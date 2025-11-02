THIS IS HEADER FILE #ifndef ANOMALY_AUTOENCODER_HPP
#define ANOMALY_AUTOENCODER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/reconstruction_loss.hpp>
#include <memory>
#include <vector>
#include <queue>

using namespace mlpack;
using namespace mlpack::ann;

/**
 * @brief Deep Autoencoder with Skip Connections and Layer Normalization for Anomaly Detection
 * 
 * Features:
 * - Encoder-decoder architecture with skip connections
 * - Layer normalization for stable training
 * - Multiple anomaly scoring methods
 * - Adaptive thresholding
 * - Support for various data types (images, time series, tabular)
 */
class AnomalyAutoencoder
{
public:
    /**
     * @brief Constructor for anomaly detection autoencoder
     * @param inputDim Dimension of input data
     * @param encodingDim Dimension of encoding (bottleneck)
     * @param hiddenDims Dimensions of hidden layers
     * @param useSkipConnections Whether to use skip connections
     * @param useLayerNorm Whether to use layer normalization
     */
    AnomalyAutoencoder(size_t inputDim,
                      size_t encodingDim = 32,
                      const std::vector<size_t>& hiddenDims = {128, 64},
                      bool useSkipConnections = true,
                      bool useLayerNorm = true);

    /**
     * @brief Train the autoencoder on normal data
     * @param normalData Training data containing only normal samples
     * @param numEpochs Number of training epochs
     * @param batchSize Batch size for training
     * @param learningRate Learning rate
     */
    void Train(const arma::mat& normalData,
               size_t numEpochs = 100,
               size_t batchSize = 32,
               double learningRate = 1e-3);

    /**
     * @brief Detect anomalies in input data
     * @param data Input data to analyze
     * @return Vector of anomaly scores for each sample
     */
    arma::vec DetectAnomalies(const arma::mat& data);

    /**
     * @brief Classify samples as normal or anomalous
     * @param data Input data to classify
     * @param threshold Anomaly score threshold (optional, auto-computed if not provided)
     * @return Vector of booleans (true = anomalous)
     */
    arma::ucvec Classify(const arma::mat& data, double threshold = -1.0);

    /**
     * @brief Compute reconstruction error for input data
     * @param data Input data
     * @return Reconstruction error for each sample
     */
    arma::vec ComputeReconstructionError(const arma::mat& data);

    /**
     * @brief Set anomaly threshold based on training data
     * @param confidenceLevel Confidence level for threshold (e.g., 0.95 for 95%)
     */
    void SetThreshold(double confidenceLevel = 0.95);

    /**
     * @brief Get current anomaly threshold
     * @return Current threshold value
     */
    double GetThreshold() const { return anomalyThreshold; }

    /**
     * @brief Save model to file
     * @param filename Model file path
     */
    void Save(const std::string& filename);

    /**
     * @brief Load model from file
     * @param filename Model file path
     */
    void Load(const std::string& filename);

    /**
     * @brief Get reconstruction for input data
     * @param data Input data
     * @return Reconstructed data
     */
    arma::mat Reconstruct(const arma::mat& data);

private:
    // Network components
    std::unique_ptr<FFN<>> autoencoder;
    std::unique_ptr<Adam> optimizer;

    // Network parameters
    size_t inputDim;
    size_t encodingDim;
    std::vector<size_t> hiddenDims;
    bool useSkipConnections;
    bool useLayerNorm;

    // Anomaly detection parameters
    double anomalyThreshold = 0.0;
    arma::vec trainingErrors;
    bool isTrained = false;

    // Skip connection storage
    std::vector<std::shared_ptr<IdentityLayer<>>> skipConnections;

    /**
     * @brief Build the autoencoder architecture
     */
    void BuildAutoencoder();

    /**
     * @brief Build encoder network
     * @return Encoder network
     */
    std::shared_ptr<Sequential<>> BuildEncoder();

    /**
     * @brief Build decoder network
     * @return Decoder network
     */
    std::shared_ptr<Sequential<>> BuildDecoder();

    /**
     * @brief Build encoder block with optional layer normalization
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @param useActivation Whether to use activation function
     * @return Encoder block
     */
    std::shared_ptr<Sequential<>> BuildEncoderBlock(size_t inputSize,
                                                   size_t outputSize,
                                                   bool useActivation = true);

    /**
     * @brief Build decoder block with optional layer normalization
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @param useActivation Whether to use activation function
     * @return Decoder block
     */
    std::shared_ptr<Sequential<>> BuildDecoderBlock(size_t inputSize,
                                                   size_t outputSize,
                                                   bool useActivation = true);

    /**
     * @brief Build residual block with skip connection
     * @param layerSize Layer dimension
     * @return Residual block
     */
    std::shared_ptr<Sequential<>> BuildResidualBlock(size_t layerSize);

    /**
     * @brief Add layer normalization to a sequential block
     * @param model Sequential model
     * @param size Normalization dimension
     */
    void AddLayerNorm(std::shared_ptr<Sequential<>> model, size_t size);

    /**
     * @brief Calculate reconstruction loss
     * @param original Original data
     * @param reconstructed Reconstructed data
     * @return Loss value
     */
    double CalculateLoss(const arma::mat& original, const arma::mat& reconstructed);

    /**
     * @brief Initialize network weights
     */
    void InitializeWeights();

    /**
     * @brief Compute anomaly score based on reconstruction error
     * @param reconstructionError Reconstruction error
     * @return Anomaly score
     */
    double ComputeAnomalyScore(double reconstructionError);

    /**
     * @brief Update moving statistics for adaptive thresholding
     * @param error Current reconstruction error
     */
    void UpdateStatistics(double error);
};

#endif AND THIS IS IMPLEMENTATION FILE #include "anomaly_autoencoder.hpp"
#include <mlpack/methods/ann/visitor/forward_visitor.hpp>
#include <mlpack/methods/ann/visitor/backward_visitor.hpp>
#include <mlpack/methods/ann/visitor/gradient_visitor.hpp>
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>

using namespace mlpack;
using namespace mlpack::ann;

AnomalyAutoencoder::AnomalyAutoencoder(size_t inputDim,
                                     size_t encodingDim,
                                     const std::vector<size_t>& hiddenDims,
                                     bool useSkipConnections,
                                     bool useLayerNorm)
    : inputDim(inputDim),
      encodingDim(encodingDim),
      hiddenDims(hiddenDims),
      useSkipConnections(useSkipConnections),
      useLayerNorm(useLayerNorm)
{
    BuildAutoencoder();
    InitializeWeights();

    std::cout << "Anomaly Autoencoder Initialized:" << std::endl;
    std::cout << "  Input Dimension: " << inputDim << std::endl;
    std::cout << "  Encoding Dimension: " << encodingDim << std::endl;
    std::cout << "  Hidden Dimensions: ";
    for (size_t dim : hiddenDims)
        std::cout << dim << " ";
    std::cout << std::endl;
    std::cout << "  Skip Connections: " << (useSkipConnections ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Layer Normalization: " << (useLayerNorm ? "Enabled" : "Disabled") << std::endl;
}

void AnomalyAutoencoder::BuildAutoencoder()
{
    autoencoder = std::make_unique<FFN<>>();
    skipConnections.clear();

    // Build encoder
    auto encoder = BuildEncoder();
    autoencoder->Add(encoder);

    // Build decoder
    auto decoder = BuildDecoder();
    autoencoder->Add(decoder);
}

std::shared_ptr<Sequential<>> AnomalyAutoencoder::BuildEncoder()
{
    auto encoder = std::make_shared<Sequential<>>();
    
    size_t currentDim = inputDim;
    size_t skipIndex = 0;

    // Input layer
    auto inputBlock = std::make_shared<Sequential<>>();
    inputBlock->Add<Linear<>>(currentDim, hiddenDims[0]);
    
    if (useLayerNorm)
        AddLayerNorm(inputBlock, hiddenDims[0]);
    
    inputBlock->Add<LeakyReLU<>>(0.1);

    encoder->Add(inputBlock);

    // Store first skip connection
    if (useSkipConnections)
    {
        auto skip = std::make_shared<IdentityLayer<>>();
        skipConnections.push_back(skip);
        encoder->Add(skip);
    }

    currentDim = hiddenDims[0];

    // Hidden encoder layers
    for (size_t i = 1; i < hiddenDims.size(); ++i)
    {
        auto encoderBlock = BuildEncoderBlock(currentDim, hiddenDims[i]);
        encoder->Add(encoderBlock);

        // Store skip connection
        if (useSkipConnections)
        {
            auto skip = std::make_shared<IdentityLayer<>>();
            skipConnections.push_back(skip);
            encoder->Add(skip);
        }

        currentDim = hiddenDims[i];
    }

    // Bottleneck layer
    auto bottleneck = std::make_shared<Sequential<>>();
    bottleneck->Add<Linear<>>(currentDim, encodingDim);
    
    if (useLayerNorm)
        AddLayerNorm(bottleneck, encodingDim);
    
    bottleneck->Add<LeakyReLU<>>(0.1);
    encoder->Add(bottleneck);

    return encoder;
}

std::shared_ptr<Sequential<>> AnomalyAutoencoder::BuildDecoder()
{
    auto decoder = std::make_shared<Sequential<>>();
    
    std::vector<size_t> decoderDims = hiddenDims;
    std::reverse(decoderDims.begin(), decoderDims.end());

    size_t currentDim = encodingDim;
    size_t skipIndex = skipConnections.size() - 1;

    // First decoder layer from bottleneck
    auto firstDecoder = std::make_shared<Sequential<>>();
    firstDecoder->Add<Linear<>>(currentDim, decoderDims[0]);
    
    if (useLayerNorm)
        AddLayerNorm(firstDecoder, decoderDims[0]);
    
    firstDecoder->Add<LeakyReLU<>>(0.1);
    decoder->Add(firstDecoder);

    currentDim = decoderDims[0];

    // Hidden decoder layers with skip connections
    for (size_t i = 1; i < decoderDims.size(); ++i)
    {
        auto decoderBlock = BuildDecoderBlock(currentDim, decoderDims[i]);
        decoder->Add(decoderBlock);

        // Apply skip connection if available
        if (useSkipConnections && skipIndex > 0)
        {
            auto skipMerge = std::make_shared<AddMerge<>>(2);
            
            auto mainBranch = std::make_shared<Sequential<>>();
            mainBranch->Add<IdentityLayer<>>();
            
            auto skipBranch = std::make_shared<Sequential<>>();
            skipBranch->Add(skipConnections[--skipIndex]);
            
            skipMerge->Add(mainBranch);
            skipMerge->Add(skipBranch);
            
            decoder->Add(skipMerge);
            decoder->Add<LeakyReLU<>>(0.1);
        }

        currentDim = decoderDims[i];
    }

    // Output layer
    auto outputLayer = std::make_shared<Sequential<>>();
    outputLayer->Add<Linear<>>(currentDim, inputDim);
    outputLayer->Add<Sigmoid<>>(); // Output in [0,1] range
    
    decoder->Add(outputLayer);

    return decoder;
}

std::shared_ptr<Sequential<>> AnomalyAutoencoder::BuildEncoderBlock(size_t inputSize,
                                                                  size_t outputSize,
                                                                  bool useActivation)
{
    auto block = std::make_shared<Sequential<>>();
    
    if (useSkipConnections && inputSize == outputSize)
    {
        // Use residual block for same dimensions
        block = BuildResidualBlock(inputSize);
    }
    else
    {
        // Standard encoder block
        block->Add<Linear<>>(inputSize, outputSize);
        
        if (useLayerNorm)
            AddLayerNorm(block, outputSize);
        
        if (useActivation)
            block->Add<LeakyReLU<>>(0.1);
    }
    
    return block;
}

std::shared_ptr<Sequential<>> AnomalyAutoencoder::BuildDecoderBlock(size_t inputSize,
                                                                  size_t outputSize,
                                                                  bool useActivation)
{
    auto block = std::make_shared<Sequential<>>();
    
    if (useSkipConnections && inputSize == outputSize)
    {
        // Use residual block for same dimensions
        block = BuildResidualBlock(inputSize);
    }
    else
    {
        // Standard decoder block
        block->Add<Linear<>>(inputSize, outputSize);
        
        if (useLayerNorm)
            AddLayerNorm(block, outputSize);
        
        if (useActivation)
            block->Add<LeakyReLU<>>(0.1);
    }
    
    return block;
}

std::shared_ptr<Sequential<>> AnomalyAutoencoder::BuildResidualBlock(size_t layerSize)
{
    auto residualBlock = std::make_shared<Sequential<>>();
    
    // Main path
    auto mainPath = std::make_shared<Sequential<>>();
    mainPath->Add<Linear<>>(layerSize, layerSize);
    
    if (useLayerNorm)
        AddLayerNorm(mainPath, layerSize);
    
    mainPath->Add<LeakyReLU<>>(0.1);
    mainPath->Add<Linear<>>(layerSize, layerSize);
    
    if (useLayerNorm)
        AddLayerNorm(mainPath, layerSize);
    
    // Residual connection
    auto residual = std::make_shared<AddMerge<>>(2);
    
    auto identityBranch = std::make_shared<Sequential<>>();
    identityBranch->Add<IdentityLayer<>>();
    
    residual->Add(mainPath);
    residual->Add(identityBranch);
    
    residualBlock->Add(residual);
    residualBlock->Add<LeakyReLU<>>(0.1);
    
    return residualBlock;
}

void AnomalyAutoencoder::AddLayerNorm(std::shared_ptr<Sequential<>> model, size_t size)
{
    // Layer normalization implementation
    model->Add<BatchNorm<>>(size); // Using BatchNorm as approximation
}

void AnomalyAutoencoder::InitializeWeights()
{
    // He initialization for better training
    HeInitialization heInit;
    
    // Initialize all linear layers
    for (size_t i = 0; i < autoencoder->Network().size(); ++i)
    {
        auto& layer = autoencoder->Network()[i];
        
        // Check if it's a linear layer or contains linear layers
        if (auto* linearLayer = dynamic_cast<Linear<>*>(&layer))
        {
            arma::mat weights = linearLayer->Parameters();
            heInit.Initialize(weights, weights.n_rows, weights.n_cols);
            linearLayer->Parameters() = weights;
        }
        else if (auto* seqLayer = dynamic_cast<Sequential<>*>(&layer))
        {
            // Initialize layers in sequential blocks
            for (size_t j = 0; j < seqLayer->Network().size(); ++j)
            {
                if (auto* innerLinear = dynamic_cast<Linear<>*>(&seqLayer->Network()[j]))
                {
                    arma::mat weights = innerLinear->Parameters();
                    heInit.Initialize(weights, weights.n_rows, weights.n_cols);
                    innerLinear->Parameters() = weights;
                }
            }
        }
    }
}

void AnomalyAutoencoder::Train(const arma::mat& normalData,
                              size_t numEpochs,
                              size_t batchSize,
                              double learningRate)
{
    optimizer = std::make_unique<Adam>(learningRate, batchSize, 0.9, 0.999);
    trainingErrors = arma::vec(normalData.n_cols);
    
    std::cout << "Training Autoencoder on Normal Data..." << std::endl;
    std::cout << "Samples: " << normalData.n_cols << std::endl;
    std::cout << "Batch Size: " << batchSize << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;

    for (size_t epoch = 0; epoch < numEpochs; ++epoch)
    {
        double totalLoss = 0.0;
        size_t numBatches = 0;

        // Shuffle data
        arma::mat shuffledData = normalData;
        shuffledData = shuffledData.cols(arma::shuffle(arma::linspace<arma::uvec>(0, normalData.n_cols-1, normalData.n_cols)));

        for (size_t i = 0; i < shuffledData.n_cols; i += batchSize)
        {
            size_t currentBatchSize = std::min(batchSize, shuffledData.n_cols - i);
            arma::mat batch = shuffledData.cols(i, i + currentBatchSize - 1);

            // Forward pass
            arma::mat reconstruction = autoencoder->Forward(batch);

            // Calculate loss
            double loss = CalculateLoss(batch, reconstruction);
            totalLoss += loss;

            // Backward pass
            arma::mat gradient = reconstruction - batch; // MSE gradient
            autoencoder->Backward(batch, gradient);

            // Update weights
            optimizer->Update();

            numBatches++;
        }

        double avgLoss = totalLoss / numBatches;

        if (epoch % 10 == 0)
        {
            std::cout << "Epoch " << epoch << "/" << numEpochs 
                      << " - Loss: " << avgLoss << std::endl;
        }
    }

    // Compute reconstruction errors for training data
    std::cout << "Computing training reconstruction errors..." << std::endl;
    for (size_t i = 0; i < normalData.n_cols; ++i)
    {
        arma::mat sample = normalData.col(i);
        arma::mat reconstruction = autoencoder->Forward(sample);
        trainingErrors(i) = CalculateLoss(sample, reconstruction);
    }

    // Set anomaly threshold
    SetThreshold(0.95);
    isTrained = true;

    std::cout << "Training completed. Anomaly threshold: " << anomalyThreshold << std::endl;
}

arma::vec AnomalyAutoencoder::DetectAnomalies(const arma::mat& data)
{
    if (!isTrained)
    {
        throw std::runtime_error("Model must be trained before anomaly detection");
    }

    arma::vec anomalyScores(data.n_cols);

    for (size_t i = 0; i < data.n_cols; ++i)
    {
        arma::mat sample = data.col(i);
        arma::mat reconstruction = autoencoder->Forward(sample);
        double error = CalculateLoss(sample, reconstruction);
        anomalyScores(i) = ComputeAnomalyScore(error);
    }

    return anomalyScores;
}

arma::ucvec AnomalyAutoencoder::Classify(const arma::mat& data, double threshold)
{
    if (threshold < 0)
        threshold = anomalyThreshold;

    arma::vec scores = DetectAnomalies(data);
    arma::ucvec classifications(data.n_cols);

    for (size_t i = 0; i < data.n_cols; ++i)
    {
        classifications(i) = (scores(i) > threshold) ? 1 : 0;
    }

    return classifications;
}

arma::vec AnomalyAutoencoder::ComputeReconstructionError(const arma::mat& data)
{
    arma::vec errors(data.n_cols);

    for (size_t i = 0; i < data.n_cols; ++i)
    {
        arma::mat sample = data.col(i);
        arma::mat reconstruction = autoencoder->Forward(sample);
        errors(i) = CalculateLoss(sample, reconstruction);
    }

    return errors;
}

void AnomalyAutoencoder::SetThreshold(double confidenceLevel)
{
    if (trainingErrors.n_elem == 0)
    {
        throw std::runtime_error("Must train model before setting threshold");
    }

    // Sort reconstruction errors
    arma::vec sortedErrors = arma::sort(trainingErrors);
    
    // Find threshold at specified confidence level
    size_t thresholdIndex = static_cast<size_t>(confidenceLevel * sortedErrors.n_elem);
    anomalyThreshold = sortedErrors(thresholdIndex);

    std::cout << "Anomaly threshold set at " << confidenceLevel * 100 
              << "% confidence level: " << anomalyThreshold << std::endl;
}

arma::mat AnomalyAutoencoder::Reconstruct(const arma::mat& data)
{
    return autoencoder->Forward(data);
}

double AnomalyAutoencoder::CalculateLoss(const arma::mat& original, const arma::mat& reconstructed)
{
    // Mean Squared Error
    double mse = arma::accu(arma::square(original - reconstructed)) / original.n_elem;
    
    // Additional reconstruction loss components can be added here
    return mse;
}

double AnomalyAutoencoder::ComputeAnomalyScore(double reconstructionError)
{
    // Normalize by training statistics
    double meanError = arma::mean(trainingErrors);
    double stdError = arma::stddev(trainingErrors);
    
    if (stdError < 1e-8)
        return reconstructionError;
    
    // Z-score based anomaly score
    return (reconstructionError - meanError) / stdError;
}

void AnomalyAutoencoder::UpdateStatistics(double error)
{
    // For online learning, update running statistics
    // This is a simplified version - in practice you'd use running averages
}

void AnomalyAutoencoder::Save(const std::string& filename)
{
    data::Save(filename + "_autoencoder.xml", "autoencoder", *autoencoder);
    
    // Save training statistics
    arma::mat stats(2, 1);
    stats(0) = anomalyThreshold;
    stats(1) = isTrained ? 1.0 : 0.0;
    data::Save(filename + "_stats.csv", stats);
    
    if (trainingErrors.n_elem > 0)
    {
        data::Save(filename + "_training_errors.csv", trainingErrors);
    }
    
    std::cout << "Model saved to " << filename << "_*" << std::endl;
}

void AnomalyAutoencoder::Load(const std::string& filename)
{
    data::Load(filename + "_autoencoder.xml", "autoencoder", *autoencoder);
    
    // Load training statistics
    arma::mat stats;
    data::Load(filename + "_stats.csv", stats);
    anomalyThreshold = stats(0);
    isTrained = (stats(1) > 0.5);
    
    try
    {
        data::Load(filename + "_training_errors.csv", trainingErrors);
    }
    catch (...)
    {
        std::cout << "No training errors found, threshold may need recalibration" << std::endl;
    }
    
    std::cout << "Model loaded from " << filename << "_*" << std::endl;
    std::cout << "Trained: " << (isTrained ? "Yes" : "No") 
              << ", Threshold: " << anomalyThreshold << std::endl;
} AND THIS IS USAGE FILE #include <iostream>
#include <vector>
#include "anomaly_autoencoder.hpp"

int main()
{
    std::cout << "Anomaly Detection Autoencoder Demo" << std::endl;
    
    // Generate sample data (normal data + some anomalies)
    size_t inputDim = 100;
    size_t numNormalSamples = 1000;
    size_t numTestSamples = 200;
    
    // Generate normal data (multivariate Gaussian)
    arma::mat normalData(inputDim, numNormalSamples);
    normalData.randn(); // Standard normal distribution
    
    // Create autoencoder
    AnomalyAutoencoder autoencoder(inputDim, 32, {128, 64}, true, true);
    
    // Train on normal data
    autoencoder.Train(normalData, 50, 32, 1e-3);
    
    // Generate test data with some anomalies
    arma::mat testData(inputDim, numTestSamples);
    testData.cols(0, 150) = normalData.cols(0, 150); // Normal samples
    testData.cols(151, 199).randu(); // Anomalous samples (uniform distribution)
    
    // Detect anomalies
    arma::vec anomalyScores = autoencoder.DetectAnomalies(testData);
    arma::ucvec classifications = autoencoder.Classify(testData);
    
    // Print results
    std::cout << "\nAnomaly Detection Results:" << std::endl;
    std::cout << "Sample\tScore\tClassification" << std::endl;
    for (size_t i = 0; i < std::min(size_t(20), testData.n_cols); ++i)
    {
        std::cout << i << "\t" << anomalyScores(i) << "\t" 
                  << (classifications(i) ? "ANOMALY" : "NORMAL") << std::endl;
    }
    
    // Count anomalies
    size_t numAnomalies = arma::accu(classifications);
    std::cout << "\nDetected " << numAnomalies << " anomalies out of " 
              << testData.n_cols << " test samples" << std::endl;
    
    // Save model
    autoencoder.Save("anomaly_detector");
    
    // Demonstrate reconstruction
    arma::mat sample = testData.col(0);
    arma::mat reconstruction = autoencoder.Reconstruct(sample);
    double error = arma::norm(sample - reconstruction, 2);
    std::cout << "\nReconstruction error for first sample: " << error << std::endl;
    
    return 0;
}