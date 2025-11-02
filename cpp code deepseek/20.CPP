#ifndef SEGMENTATION_NETWORK_HPP
#define SEGMENTATION_NETWORK_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/dice_loss.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <memory>
#include <vector>

using namespace mlpack;
using namespace mlpack::ann;

/**
 * @brief Neural network with convolutional and attention blocks for multi-class image segmentation
 * 
 * Features:
 * - U-Net like architecture with skip connections
 * - Multi-scale attention mechanisms
 * - Residual connections
 * - Multiple loss functions (Dice, Cross-Entropy)
 */
class SegmentationNetwork
{
public:
    /**
     * @brief Constructor for segmentation network
     * @param inputChannels Number of input image channels
     * @param numClasses Number of segmentation classes
     * @param encoderFilters Base number of filters for encoder
     * @param useAttention Whether to use attention mechanisms
     * @param attentionType Type of attention ("channel", "spatial", "dual")
     */
    SegmentationNetwork(size_t inputChannels = 3,
                       size_t numClasses = 1,
                       size_t encoderFilters = 64,
                       bool useAttention = true,
                       const std::string& attentionType = "dual");

    /**
     * @brief Forward pass through the network
     * @param input Input tensor (C x H x W x N)
     * @return Segmentation mask output
     */
    arma::mat Forward(const arma::mat& input);

    /**
     * @brief Train the segmentation network
     * @param inputs Training images
     * @param targets Segmentation masks
     * @param numEpochs Number of training epochs
     * @param batchSize Batch size
     * @param learningRate Learning rate
     */
    void Train(const arma::mat& inputs,
               const arma::mat& targets,
               size_t numEpochs = 100,
               size_t batchSize = 8,
               double learningRate = 1e-4);

    /**
     * @brief Predict segmentation masks for input images
     * @param inputs Input images
     * @return Predicted segmentation masks
     */
    arma::mat Predict(const arma::mat& inputs);

    /**
     * @brief Calculate evaluation metrics
     * @param predictions Network predictions
     * @param targets Ground truth masks
     * @return Dictionary of metrics (Dice, IoU, Accuracy)
     */
    std::map<std::string, double> Evaluate(const arma::mat& predictions,
                                          const arma::mat& targets);

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

private:
    // Network components
    std::unique_ptr<FFN<>> network;
    std::unique_ptr<Adam> optimizer;

    // Network parameters
    size_t inputChannels;
    size_t numClasses;
    size_t encoderFilters;
    bool useAttention;
    std::string attentionType;

    // Encoder and decoder feature maps for skip connections
    std::vector<arma::mat> encoderFeatures;

    /**
     * @brief Build the complete segmentation network
     */
    void BuildNetwork();

    /**
     * @brief Build encoder (contracting path)
     * @param inputFilters Number of input filters
     * @param outputFilters Number of output filters
     * @param applyAttention Whether to apply attention
     * @return Encoder block
     */
    std::shared_ptr<Sequential<>> BuildEncoderBlock(size_t inputFilters,
                                                   size_t outputFilters,
                                                   bool applyAttention = false);

    /**
     * @brief Build decoder (expanding path)
     * @param inputFilters Number of input filters
     * @param outputFilters Number of output filters
     * @param applyAttention Whether to apply attention
     * @return Decoder block
     */
    std::shared_ptr<Sequential<>> BuildDecoderBlock(size_t inputFilters,
                                                   size_t outputFilters,
                                                   bool applyAttention = false);

    /**
     * @brief Build residual block
     * @param filters Number of filters
     * @return Residual block
     */
    std::shared_ptr<Sequential<>> BuildResidualBlock(size_t filters);

    /**
     * @brief Build channel attention module (SENet-like)
     * @param filters Number of filters
     * @param reduction Reduction ratio
     * @return Channel attention module
     */
    std::shared_ptr<Sequential<>> BuildChannelAttention(size_t filters,
                                                       size_t reduction = 16);

    /**
     * @brief Build spatial attention module
     * @param filters Number of filters
     * @return Spatial attention module
     */
    std::shared_ptr<Sequential<>> BuildSpatialAttention(size_t filters);

    /**
     * @brief Build dual attention module (both channel and spatial)
     * @param filters Number of filters
     * @return Dual attention module
     */
    std::shared_ptr<Sequential<>> BuildDualAttention(size_t filters);

    /**
     * @brief Build attention gate for skip connections
     * @param filters Number of filters
     * @return Attention gate module
     */
    std::shared_ptr<Sequential<>> BuildAttentionGate(size_t filters);

    /**
     * @brief Calculate combined loss (Dice + Cross-Entropy)
     * @param predictions Network predictions
     * @param targets Ground truth masks
     * @return Combined loss value
     */
    double CalculateLoss(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Calculate Dice coefficient
     * @param predictions Network predictions
     * @param targets Ground truth masks
     * @return Dice coefficient
     */
    double CalculateDice(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Calculate Intersection over Union (IoU)
     * @param predictions Network predictions
     * @param targets Ground truth masks
     * @return IoU value
     */
    double CalculateIoU(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Initialize network weights
     */
    void InitializeWeights();

    /**
     * @brief Apply softmax to segmentation output
     * @param input Network output
     * @return Softmax probabilities
     */
    arma::mat ApplySoftmax(const arma::mat& input);
};

#endif