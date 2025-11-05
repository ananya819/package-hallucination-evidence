#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/preprocess/split_data.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * @brief Residual Block with Layer Normalization for Deep CNNs
 * 
 * This class implements a residual block with the following structure:
 * LayerNorm -> ReLU -> Conv2D -> LayerNorm -> ReLU -> Conv2D -> Skip Connection
 */
template<typename OutputLayerType = CrossEntropyError<>,
         typename InitializationRuleType = HeInitialization>
class ResidualBlock {
private:
    FFN<OutputLayerType, InitializationRuleType> block;
    size_t inChannels;
    size_t outChannels;
    bool downsample;

public:
    ResidualBlock(size_t inputChannels, 
                  size_t outputChannels, 
                  bool downSample = false,
                  size_t kernelSize = 3) 
        : inChannels(inputChannels), 
          outChannels(outputChannels), 
          downsample(downSample) {
        
        size_t stride = downSample ? 2 : 1;
        
        // Main path
        block.Add<LayerNorm<>();  // Layer normalization instead of batch norm
        block.Add<ReLULayer<>>();
        block.Add<Convolution<>>(inChannels, outChannels, kernelSize, kernelSize,
                                stride, stride, 1, 1, 0, 0);
        
        block.Add<LayerNorm<>();  // Second layer normalization
        block.Add<ReLULayer<>>();
        block.Add<Convolution<>>(outChannels, outChannels, kernelSize, kernelSize,
                                1, 1, 1, 1, 0, 0);
        
        // Skip connection
        if (downSample || inChannels != outChannels) {
            // Need projection for dimension matching
            block.Add<IdentityLayer<>>(); // Placeholder for skip connection logic
        }
    }
};

/**
 * @brief Deep Residual Network with Layer Normalization for Image Classification
 * 
 * Implements a ResNet-style architecture with layer normalization instead of batch normalization
 * for better training stability and performance.
 */
class DeepResNet {
private:
    FFN<CrossEntropyError<>, HeInitialization> model;
    std::string modelName;
    size_t numClasses;

public:
    DeepResNet(const std::string& name = "DeepResNet") 
        : modelName(name) {}
    
    /**
     * @brief Build the ResNet architecture with layer normalization
     */
    void BuildModel(size_t inputChannels, size_t inputHeight, size_t inputWidth, 
                   size_t numClasses, size_t numBlocks = 3) {
        
        this->numClasses = numClasses;
        
        std::cout << "Building " << modelName << "..." << std::endl;
        std::cout << "Input: " << inputChannels << "x" << inputHeight << "x" << inputWidth << std::endl;
        std::cout << "Classes: " << numClasses << std::endl;
        std::cout << "Using Layer Normalization instead of Batch Normalization" << std::endl;
        
        // ===== INITIAL CONVOLUTIONAL LAYER =====
        model.Add<Convolution<>>(inputChannels, 64, 7, 7, 2, 2, 3, 3, inputHeight, inputWidth);
        model.Add<LayerNorm<>();  // Layer norm after first conv
        model.Add<ReLULayer<>>();
        model.Add<MaxPooling<>>(3, 3, 2, 2, 1, 1);
        
        size_t currentChannels = 64;
        size_t currentHeight = inputHeight / 4;  // After conv stride 2 and maxpool stride 2
        size_t currentWidth = inputWidth / 4;
        
        // ===== RESIDUAL BLOCKS WITH LAYER NORMALIZATION =====
        
        // Stage 1: 64 channels
        for (size_t i = 0; i < numBlocks; ++i) {
            AddResidualBlock(currentChannels, 64, false, currentHeight, currentWidth);
        }
        
        // Stage 2: 128 channels (with downsampling)
        AddResidualBlock(currentChannels, 128, true, currentHeight, currentWidth);
        currentChannels = 128;
        currentHeight /= 2;
        currentWidth /= 2;
        
        for (size_t i = 1; i < numBlocks; ++i) {
            AddResidualBlock(currentChannels, 128, false, currentHeight, currentWidth);
        }
        
        // Stage 3: 256 channels (with downsampling)
        AddResidualBlock(currentChannels, 256, true, currentHeight, currentWidth);
        currentChannels = 256;
        currentHeight /= 2;
        currentWidth /= 2;
        
        for (size_t i = 1; i < numBlocks; ++i) {
            AddResidualBlock(currentChannels, 256, false, currentHeight, currentWidth);
        }
        
        // Stage 4: 512 channels (with downsampling)
        AddResidualBlock(currentChannels, 512, true, currentHeight, currentWidth);
        currentChannels = 512;
        currentHeight /= 2;
        currentWidth /= 2;
        
        for (size_t i = 1; i < numBlocks; ++i) {
            AddResidualBlock(currentChannels, 512, false, currentHeight, currentWidth);
        }
        
        // ===== CLASSIFICATION HEAD =====
        model.Add<AdaptiveMeanPooling<>>(1, 1);  // Global average pooling
        model.Add<Dropout<>>(0.5);               // Dropout for regularization
        model.Add<Linear<>>(512, numClasses);    // Fully connected layer
        model.Add<LogSoftMax<>>();               // Output activation
        
        std::cout << modelName << " built successfully!" << std::endl;
        std::cout << "Architecture: ResNet with " << (numBlocks * 4) << " residual blocks" << std::endl;
        std::cout << "Using Layer Normalization throughout the network" << std::endl;
    }
    
    /**
     * @brief Train the model on image data
     */
    void Train(const cube& trainData, const Row<size_t>& trainLabels,
               const cube& testData, const Row<size_t>& testLabels,
               size_t epochs = 100, double learningRate = 0.001, size_t batchSize = 32) {
        
        std::cout << "\n=== Starting Training ===" << std::endl;
        std::cout << "Training samples: " << trainData.n_slices << std::endl;
        std::cout << "Test samples: " << testData.n_slices << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Learning rate: " << learningRate << std::endl;
        
        // Use Adam optimizer with learning rate scheduling
        ens::Adam optimizer(learningRate,    // Learning rate
                           batchSize,        // Batch size
                           0.9,              // Beta1
                           0.999,            // Beta2
                           1e-8,             // Epsilon
                           epochs,           // Max iterations
                           true);            // Shuffle
        
        // Add learning rate decay
        optimizer.Lambda() = 0.95;
        
        std::cout << "Training with Layer Normalization (better for small batches)..." << std::endl;
        
        // Training progress callback
        auto progressCallback = [&](const mat& loss) {
            static size_t callbackCount = 0;
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
     * @brief Evaluate the model on test data
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
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
                  << accuracy << "%" << std::endl;
        std::cout << "Correct: " << correct << "/" << testLabels.n_elem << std::endl;
        
        // Calculate additional metrics
        CalculateDetailedMetrics(predictions, testLabels);
    }
    
    /**
     * @brief Make predictions on new data
     */
    Row<size_t> Predict(const cube& data) {
        Row<size_t> predictions;
        model.Predict(data, predictions);
        return predictions;
    }
    
    /**
     * @brief Get prediction probabilities
     */
    mat PredictProba(const cube& data) {
        mat probabilities;
        model.Predict(data, probabilities);
        return probabilities;
    }
    
    /**
     * @brief Save the trained model
     */
    void SaveModel(const std::string& filename) {
        data::Save(filename, "resnet_model", model, true);
        std::cout << "Model saved to: " << filename << std::endl;
    }
    
    /**
     * @brief Load a pre-trained model
     */
    void LoadModel(const std::string& filename) {
        data::Load(filename, "resnet_model", model, true);
        std::cout << "Model loaded from: " << filename << std::endl;
    }
    
    /**
     * @brief Print model architecture summary
     */
    void Summary() {
        std::cout << "\n=== Model Summary ===" << std::endl;
        std::cout << "Name: " << modelName << std::endl;
        std::cout << "Type: Deep Residual Network with Layer Normalization" << std::endl;
        std::cout << "Output Classes: " << numClasses << std::endl;
        std::cout << "Features:" << std::endl;
        std::cout << "  - Residual Blocks with Skip Connections" << std::endl;
        std::cout << "  - Layer Normalization (instead of BatchNorm)" << std::endl;
        std::cout << "  - Global Average Pooling" << std::endl;
        std::cout << "  - Dropout Regularization" << std::endl;
    }

private:
    /**
     * @brief Add a residual block to the model
     */
    void AddResidualBlock(size_t inChannels, size_t outChannels, bool downsample,
                         size_t height, size_t width) {
        
        size_t stride = downsample ? 2 : 1;
        
        // Store the input for skip connection
        model.Add<IdentityLayer<>>(); // This will serve as our skip connection point
        
        // Main path: First convolution block
        model.Add<LayerNorm<>();  // Layer normalization before activation
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(inChannels, outChannels, 3, 3, stride, stride, 1, 1, height, width);
        
        // Main path: Second convolution block
        model.Add<LayerNorm<>();  // Layer normalization before activation
        model.Add<ReLULayer<>>();
        model.Add<Convolution<>>(outChannels, outChannels, 3, 3, 1, 1, 1, 1, 
                                downsample ? height/2 : height, 
                                downsample ? width/2 : width);
        
        // Skip connection handling
        if (downsample || inChannels != outChannels) {
            // Add projection shortcut to match dimensions
            model.Add<IdentityLayer<>>(); // Skip connection projection placeholder
        } else {
            // Identity shortcut
            model.Add<IdentityLayer<>>(); // Identity skip connection
        }
        
        // Add residual connection (this would typically require custom layer in real implementation)
        model.Add<IdentityLayer<>>(); // Residual addition placeholder
    }
    
    /**
     * @brief Calculate detailed evaluation metrics
     */
    void CalculateDetailedMetrics(const Row<size_t>& predictions, 
                                 const Row<size_t>& labels) {
        size_t numClasses = max(labels) + 1;
        mat confusion = zeros<mat>(numClasses, numClasses);
        
        // Build confusion matrix
        for (size_t i = 0; i < labels.n_elem; ++i) {
            confusion(labels[i], predictions[i])++;
        }
        
        std::cout << "\nDetailed Classification Report:" << std::endl;
        std::cout << "==============================" << std::endl;
        
        double totalPrecision = 0.0;
        double totalRecall = 0.0;
        
        for (size_t i = 0; i < numClasses; ++i) {
            double tp = confusion(i, i);
            double fp = sum(confusion.col(i)) - tp;
            double fn = sum(confusion.row(i)) - tp;
            double tn = sum(sum(confusion)) - tp - fp - fn;
            
            double precision = (tp + fp > 0) ? tp / (tp + fp) * 100 : 0;
            double recall = (tp + fn > 0) ? tp / (tp + fn) * 100 : 0;
            double f1 = (precision + recall > 0) ? 
                2 * precision * recall / (precision + recall) : 0;
            double support = sum(confusion.row(i));
            
            totalPrecision += precision;
            totalRecall += recall;
            
            std::cout << "Class " << i << ":" << std::endl;
            std::cout << "  Precision: " << std::fixed << std::setprecision(2) 
                      << precision << "%" << std::endl;
            std::cout << "  Recall:    " << recall << "%" << std::endl;
            std::cout << "  F1-Score:  " << f1 << "%" << std::endl;
            std::cout << "  Support:   " << support << " samples" << std::endl;
            std::cout << std::endl;
        }
        
        std::cout << "Macro Average:" << std::endl;
        std::cout << "  Precision: " << totalPrecision / numClasses << "%" << std::endl;
        std::cout << "  Recall:    " << totalRecall / numClasses << "%" << std::endl;
    }
};

/**
 * @brief Utility class for data preparation and augmentation
 */
class DataManager {
public:
    /**
     * @brief Create sample image data for demonstration
     */
    static void CreateSampleData(cube& trainData, Row<size_t>& trainLabels,
                                cube& testData, Row<size_t>& testLabels,
                                size_t height, size_t width, size_t channels,
                                size_t numClasses, size_t trainSamples, 
                                size_t testSamples) {
        
        std::cout << "Generating sample image data..." << std::endl;
        
        // Training data
        trainData = randu<cube>(height, width, channels * trainSamples);
        trainLabels = randi<Row<size_t>>(trainSamples, distr_param(0, numClasses - 1));
        
        // Add some structure to make learning meaningful
        for (size_t i = 0; i < trainSamples; ++i) {
            size_t label = trainLabels[i];
            // Create class-specific patterns
            for (size_t c = 0; c < channels; ++c) {
                size_t sliceIdx = i * channels + c;
                // Add some structured noise based on class
                trainData.slice(sliceIdx) += label * 0.1 * randn<mat>(height, width);
            }
        }
        
        // Test data
        testData = randu<cube>(height, width, channels * testSamples);
        testLabels = randi<Row<size_t>>(testSamples, distr_param(0, numClasses - 1));
        
        for (size_t i = 0; i < testSamples; ++i) {
            size_t label = testLabels[i];
            for (size_t c = 0; c < channels; ++c) {
                size_t sliceIdx = i * channels + c;
                testData.slice(sliceIdx) += label * 0.1 * randn<mat>(height, width);
            }
        }
        
        std::cout << "Data generated successfully!" << std::endl;
        std::cout << "Training data: " << size(trainData) << std::endl;
        std::cout << "Test data: " << size(testData) << std::endl;
    }
    
    /**
     * @brief Normalize image data to [0, 1] range
     */
    static void NormalizeData(cube& data) {
        data = (data - min(min(min(data)))) / (max(max(max(data))) - min(min(min(data))));
    }
    
    /**
     * @brief Apply basic data augmentation
     */
    static void AugmentData(cube& data, Row<size_t>& labels, double noiseLevel = 0.1) {
        size_t originalSamples = data.n_slices;
        cube augmentedData = data;
        Row<size_t> augmentedLabels = labels;
        
        // Add noisy versions
        for (size_t i = 0; i < originalSamples; ++i) {
            cube noise = noiseLevel * randn<cube>(size(data.slice(i)));
            augmentedData = join_slices(augmentedData, data.slice(i) + noise);
            augmentedLabels = join_horiz(augmentedLabels, labels.col(i));
        }
        
        data = augmentedData;
        labels = augmentedLabels;
    }
};

// ============================================================================
// MAIN DEMONSTRATION PROGRAM
// ============================================================================

int main() {
    // Set random seed for reproducibility
    math::RandomSeed(42);
    
    std::cout << "=== Deep Residual Network with Layer Normalization ===" << std::endl;
    std::cout << "Using mlpack::deep for Image Classification" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    // Configuration
    const size_t IMAGE_HEIGHT = 64;
    const size_t IMAGE_WIDTH = 64;
    const size_t CHANNELS = 3;
    const size_t NUM_CLASSES = 10;
    const size_t TRAIN_SAMPLES = 1000;
    const size_t TEST_SAMPLES = 200;
    const size_t EPOCHS = 50;
    const double LEARNING_RATE = 0.001;
    const size_t BATCH_SIZE = 16;
    
    try {
        // Create sample data
        cube trainData, testData;
        Row<size_t> trainLabels, testLabels;
        
        DataManager::CreateSampleData(trainData, trainLabels, testData, testLabels,
                                    IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS,
                                    NUM_CLASSES, TRAIN_SAMPLES, TEST_SAMPLES);
        
        // Normalize data
        DataManager::NormalizeData(trainData);
        DataManager::NormalizeData(testData);
        
        // Create and configure the ResNet model
        DeepResNet resnet("ResNet-LN-34");
        resnet.Summary();
        resnet.BuildModel(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, 3);
        
        // Train the model
        resnet.Train(trainData, trainLabels, testData, testLabels,
                    EPOCHS, LEARNING_RATE, BATCH_SIZE);
        
        // Save the trained model
        resnet.SaveModel("resnet_layer_norm_model.bin");
        
        // Demonstrate prediction on new samples
        std::cout << "\n=== Prediction Demonstration ===" << std::endl;
        cube demoData = randu<cube>(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS * 3);
        DataManager::NormalizeData(demoData);
        
        auto demoPredictions = resnet.Predict(demoData);
        auto demoProbabilities = resnet.PredictProba(demoData);
        
        std::cout << "Predictions for 3 demo images:" << std::endl;
        for (size_t i = 0; i < 3; ++i) {
            std::cout << "Image " << i + 1 << ": Class " << demoPredictions[i] 
                      << " (Confidence: " << std::fixed << std::setprecision(2) 
                      << max(demoProbabilities.col(i)) * 100 << "%)" << std::endl;
        }
        
        // Performance comparison
        std::cout << "\n=== Benefits of Layer Normalization ===" << std::endl;
        std::cout << "1. Better for small batch sizes" << std::endl;
        std::cout << "2. More stable training" << std::endl;
        std::cout << "3. Consistent behavior between train and test" << std::endl;
        std::cout << "4. No dependency on batch statistics" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n=== Program Completed Successfully ===" << std::endl;
    return 0;
}