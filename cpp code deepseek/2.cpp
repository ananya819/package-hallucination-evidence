#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/preprocess/split_data.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * @brief Deep CNN Classifier using mlpack for image classification
 * 
 * This class implements a deep convolutional neural network similar to VGG architecture
 * for image classification tasks. It includes convolutional layers, pooling, dropout,
 * and fully connected layers.
 */
class DeepCNNClassifier {
private:
    FFN<CrossEntropyError<>, HeInitialization> model;
    std::string modelName;

public:
    DeepCNNClassifier(const std::string& name = "DeepCNN") : modelName(name) {}
    
    /**
     * @brief Builds the deep CNN architecture
     * @param inputChannels Number of input channels (1 for grayscale, 3 for RGB)
     * @param inputWidth Width of input images
     * @param inputHeight Height of input images
     * @param numClasses Number of output classes
     */
    void BuildModel(int inputChannels, int inputWidth, int inputHeight, int numClasses) {
        std::cout << "Building " << modelName << "..." << std::endl;
        std::cout << "Input: " << inputChannels << "x" << inputWidth << "x" << inputHeight << std::endl;
        std::cout << "Output: " << numClasses << " classes" << std::endl;
        
        // ===== CONVOLUTIONAL BLOCKS =====
        
        // Block 1: Two 3x3 conv layers + max pooling
        model.Add<Convolution<>>(inputChannels, 64, 3, 3, 1, 1, 1, 1, 
                                inputWidth, inputHeight);
        model.Add<BatchNorm<>>(64);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(64, 64, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight);
        model.Add<BatchNorm<>>(64);
        model.Add<ReLULayer<>>();
        model.Add<MaxPooling<>>(2, 2, 2, 2, true);
        int currentWidth = inputWidth / 2;
        int currentHeight = inputHeight / 2;
        
        // Block 2: Two 3x3 conv layers + max pooling
        model.Add<Convolution<>>(64, 128, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(128);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(128, 128, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(128);
        model.Add<ReLULayer<>>();
        model.Add<MaxPooling<>>(2, 2, 2, 2, true);
        currentWidth /= 2;
        currentHeight /= 2;
        
        // Block 3: Three 3x3 conv layers + max pooling
        model.Add<Convolution<>>(128, 256, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(256);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(256, 256, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(256);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(256, 256, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(256);
        model.Add<ReLULayer<>>();
        model.Add<MaxPooling<>>(2, 2, 2, 2, true);
        currentWidth /= 2;
        currentHeight /= 2;
        
        // Block 4: Three 3x3 conv layers + max pooling
        model.Add<Convolution<>>(256, 512, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(512);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(512, 512, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(512);
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(512, 512, 3, 3, 1, 1, 1, 1, currentWidth, currentHeight);
        model.Add<BatchNorm<>>(512);
        model.Add<ReLULayer<>>();
        model.Add<MaxPooling<>>(2, 2, 2, 2, true);
        currentWidth /= 2;
        currentHeight /= 2;
        
        // ===== FULLY CONNECTED LAYERS =====
        
        // Calculate flattened size
        int flattenedSize = 512 * currentWidth * currentHeight;
        std::cout << "Flattened size: " << flattenedSize << std::endl;
        
        // Fully connected layers with dropout
        model.Add<Linear<>>(flattenedSize, 4096);
        model.Add<ReLULayer<>>();
        model.Add<Dropout<>>(0.5);
        
        model.Add<Linear<>>(4096, 4096);
        model.Add<ReLULayer<>>();
        model.Add<Dropout<>>(0.5);
        
        model.Add<Linear<>>(4096, 1024);
        model.Add<ReLULayer<>>();
        model.Add<Dropout<>>(0.3);
        
        // Output layer
        model.Add<Linear<>>(1024, numClasses);
        model.Add<LogSoftMax<>>();
        
        std::cout << modelName << " built successfully!" << std::endl;
        std::cout << "Architecture: 13 convolutional layers + 3 fully connected layers" << std::endl;
    }
    
    /**
     * @brief Trains the CNN model
     * @param trainData Training images (cube: width x height x (channels * samples))
     * @param trainLabels Training labels
     * @param testData Test images
     * @param testLabels Test labels
     * @param epochs Number of training epochs
     * @param learningRate Initial learning rate
     * @param batchSize Batch size for training
     */
    void Train(const cube& trainData, const Row<size_t>& trainLabels,
               const cube& testData, const Row<size_t>& testLabels,
               int epochs = 50, double learningRate = 0.001, int batchSize = 32) {
        
        std::cout << "\n=== Starting Training ===" << std::endl;
        std::cout << "Training samples: " << trainData.n_slices << std::endl;
        std::cout << "Test samples: " << testData.n_slices << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Learning rate: " << learningRate << std::endl;
        
        // Use Adam optimizer with learning rate scheduling
        ens::Adam optimizer(
            learningRate,  // Learning rate
            batchSize,     // Batch size
            0.9,           // Beta1
            0.999,         // Beta2
            1e-8,          // Epsilon
            epochs,        // Max iterations
            true           // Shuffle
        );
        
        // Add learning rate decay
        optimizer.Lambda() = 0.95;
        
        // Training progress callback
        auto progressCallback = [&](const arma::mat& loss) {
            static int callbackCount = 0;
            if (++callbackCount % 10 == 0) {
                std::cout << "Epoch " << callbackCount << ", Loss: " << loss(0) << std::endl;
            }
            return false;
        };
        
        // Train the model
        model.Train(trainData, trainLabels, optimizer, progressCallback);
        
        std::cout << "Training completed!" << std::endl;
        
        // Evaluate on test set
        Evaluate(testData, testLabels);
    }
    
    /**
     * @brief Evaluates the model on test data
     * @param testData Test images
     * @param testLabels Test labels
     */
    void Evaluate(const cube& testData, const Row<size_t>& testLabels) {
        std::cout << "\n=== Model Evaluation ===" << std::endl;
        
        Row<size_t> predictions;
        model.Predict(testData, predictions);
        
        // Calculate overall accuracy
        size_t correct = 0;
        for (size_t i = 0; i < testLabels.n_elem; ++i) {
            if (predictions[i] == testLabels[i]) {
                correct++;
            }
        }
        
        double accuracy = (double)correct / testLabels.n_elem * 100.0;
        std::cout << "Overall Test Accuracy: " << accuracy << "%" << std::endl;
        std::cout << "Correct: " << correct << "/" << testLabels.n_elem << std::endl;
        
        // Calculate confusion matrix and per-class metrics
        CalculateDetailedMetrics(predictions, testLabels);
    }
    
    /**
     * @brief Makes predictions on new data
     * @param data Input images
     * @param predictions Output predictions
     */
    void Predict(const cube& data, Row<size_t>& predictions) {
        model.Predict(data, predictions);
    }
    
    /**
     * @brief Makes probability predictions on new data
     * @param data Input images
     * @param probabilities Output class probabilities
     */
    void PredictProba(const cube& data, mat& probabilities) {
        model.Predict(data, probabilities);
    }
    
    /**
     * @brief Saves the trained model to disk
     * @param filename Output filename
     */
    void SaveModel(const std::string& filename) {
        data::Save(filename, "deep_cnn_model", model, true);
        std::cout << "Model saved to: " << filename << std::endl;
    }
    
    /**
     * @brief Loads a pre-trained model from disk
     * @param filename Input filename
     */
    void LoadModel(const std::string& filename) {
        data::Load(filename, "deep_cnn_model", model, true);
        std::cout << "Model loaded from: " << filename << std::endl;
    }
    
    /**
     * @brief Returns the model summary
     */
    void Summary() {
        std::cout << "\n=== Model Summary ===" << std::endl;
        std::cout << "Name: " << modelName << std::endl;
        std::cout << "Type: Deep Convolutional Neural Network" << std::endl;
        std::cout << "Layers: 13 convolutional + 4 fully connected" << std::endl;
        std::cout << "Features: Batch Normalization, Dropout, ReLU activation" << std::endl;
    }

private:
    /**
     * @brief Calculates detailed evaluation metrics
     * @param predictions Model predictions
     * @param labels True labels
     */
    void CalculateDetailedMetrics(const Row<size_t>& predictions, 
                                 const Row<size_t>& labels) {
        size_t numClasses = max(labels) + 1;
        mat confusion = zeros<mat>(numClasses, numClasses);
        
        // Build confusion matrix
        for (size_t i = 0; i < labels.n_elem; ++i) {
            confusion(labels[i], predictions[i])++;
        }
        
        std::cout << "\nDetailed Metrics:" << std::endl;
        std::cout << "-----------------" << std::endl;
        
        for (size_t i = 0; i < numClasses; ++i) {
            double tp = confusion(i, i);
            double fp = sum(confusion.col(i)) - tp;
            double fn = sum(confusion.row(i)) - tp;
            
            double precision = (tp + fp > 0) ? tp / (tp + fp) * 100 : 0;
            double recall = (tp + fn > 0) ? tp / (tp + fn) * 100 : 0;
            double f1 = (precision + recall > 0) ? 
                2 * precision * recall / (precision + recall) : 0;
            
            std::cout << "Class " << i << ": " << std::endl;
            std::cout << "  Precision: " << precision << "%" << std::endl;
            std::cout << "  Recall: " << recall << "%" << std::endl;
            std::cout << "  F1-Score: " << f1 << "%" << std::endl;
            std::cout << "  Support: " << sum(confusion.row(i)) << std::endl;
        }
        
        std::cout << "\nConfusion Matrix:" << std::endl;
        std::cout << confusion << std::endl;
    }
};

// ============================================================================
// DATA LOADING AND PREPROCESSING UTILITIES
// ============================================================================

/**
 * @brief Creates sample image data for demonstration
 * @param trainData Output training data
 * @param trainLabels Output training labels
 * @param testData Output test data
 * @param testLabels Output test labels
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param numClasses Number of classes
 * @param trainSamples Number of training samples
 * @param testSamples Number of test samples
 */
void CreateSampleData(cube& trainData, Row<size_t>& trainLabels,
                     cube& testData, Row<size_t>& testLabels,
                     int width, int height, int channels,
                     int numClasses, int trainSamples, int testSamples) {
    std::cout << "Generating sample data..." << std::endl;
    
    // Create training data with some pattern to make learning possible
    trainData = zeros<cube>(width, height, channels * trainSamples);
    trainLabels = zeros<Row<size_t>>(trainSamples);
    
    for (size_t i = 0; i < trainSamples; ++i) {
        size_t label = i % numClasses;
        trainLabels[i] = label;
        
        // Create simple patterns for different classes
        for (int c = 0; c < channels; ++c) {
            size_t sliceIndex = i * channels + c;
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    // Create class-specific patterns
                    double value = 0.1 + 0.8 * ((x + y + label * 10) % 20) / 20.0;
                    trainData(x, y, sliceIndex) = value;
                }
            }
        }
    }
    
    // Create test data similarly
    testData = zeros<cube>(width, height, channels * testSamples);
    testLabels = zeros<Row<size_t>>(testSamples);
    
    for (size_t i = 0; i < testSamples; ++i) {
        size_t label = i % numClasses;
        testLabels[i] = label;
        
        for (int c = 0; c < channels; ++c) {
            size_t sliceIndex = i * channels + c;
            for (int x = 0; x < width; ++x) {
                for (int y = 0; y < height; ++y) {
                    double value = 0.1 + 0.8 * ((x + y + label * 10 + 5) % 20) / 20.0;
                    testData(x, y, sliceIndex) = value;
                }
            }
        }
    }
    
    std::cout << "Sample data generated successfully!" << std::endl;
    std::cout << "Training data shape: " << size(trainData) << std::endl;
    std::cout << "Test data shape: " << size(testData) << std::endl;
}

// ============================================================================
// MAIN FUNCTION WITH DEMONSTRATION
// ============================================================================

int main() {
    // Set random seed for reproducibility
    math::RandomSeed(42);
    
    std::cout << "=== Deep CNN Image Classification with mlpack ===" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Configuration
    const int IMAGE_WIDTH = 64;    // Using smaller images for demonstration
    const int IMAGE_HEIGHT = 64;
    const int CHANNELS = 3;        // RGB images
    const int NUM_CLASSES = 5;
    const int TRAIN_SAMPLES = 500;
    const int TEST_SAMPLES = 100;
    const int EPOCHS = 30;
    const double LEARNING_RATE = 0.001;
    const int BATCH_SIZE = 16;
    
    try {
        // Create sample data
        cube trainData, testData;
        Row<size_t> trainLabels, testLabels;
        
        CreateSampleData(trainData, trainLabels, testData, testLabels,
                        IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS,
                        NUM_CLASSES, TRAIN_SAMPLES, TEST_SAMPLES);
        
        // Create and build the deep CNN model
        DeepCNNClassifier cnn("VGG-Style CNN");
        cnn.Summary();
        cnn.BuildModel(CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CLASSES);
        
        // Train the model
        cnn.Train(trainData, trainLabels, testData, testLabels,
                 EPOCHS, LEARNING_RATE, BATCH_SIZE);
        
        // Save the trained model
        cnn.SaveModel("deep_cnn_model.bin");
        
        // Demonstrate prediction on new samples
        std::cout << "\n=== Prediction Demo ===" << std::endl;
        cube demoData = randu<cube>(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS * 3);
        Row<size_t> demoPredictions;
        cnn.Predict(demoData, demoPredictions);
        
        std::cout << "Predictions for 3 demo samples: ";
        for (size_t i = 0; i < demoPredictions.n_elem; ++i) {
            std::cout << demoPredictions[i] << " ";
        }
        std::cout << std::endl;
        
        // Demonstrate probability predictions
        mat demoProbabilities;
        cnn.PredictProba(demoData, demoProbabilities);
        std::cout << "Class probabilities for first sample:" << std::endl;
        std::cout << demoProbabilities.col(0).t() << std::endl;
        
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n=== Program Completed Successfully ===" << std::endl;
    return 0;
}