#ifndef ATTENTION_SRGAN_HPP
#define ATTENTION_SRGAN_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/loss_functions.hpp>
#include <memory>
#include <vector>

using namespace mlpack;
using namespace mlpack::ann;

/**
 * @brief Attention-guided Generative Adversarial Network for Image Super-Resolution
 * 
 * This implementation includes:
 * - Generator with channel and spatial attention mechanisms
 * - Discriminator with spectral normalization
 * - Feature extraction for perceptual loss
 * - Multi-scale training support
 */
class AttentionSRGAN
{
public:
    /**
     * @brief Constructor for AttentionSRGAN
     * @param scaleFactor Super-resolution scale factor (2, 4, 8)
     * @param imgChannels Number of image channels (3 for RGB, 1 for grayscale)
     * @param attentionType Type of attention mechanism ("channel", "spatial", "dual")
     */
    AttentionSRGAN(int scaleFactor = 4, 
                   int imgChannels = 3,
                   const std::string& attentionType = "dual");

    /**
     * @brief Train the SRGAN model
     * @param lrImages Low-resolution training images
     * @param hrImages High-resolution target images
     * @param numEpochs Number of training epochs
     * @param batchSize Batch size for training
     */
    void Train(const arma::mat& lrImages,
               const arma::mat& hrImages,
               int numEpochs = 100,
               int batchSize = 16);

    /**
     * @brief Generate super-resolved image
     * @param lrImage Low-resolution input image
     * @return Super-resolved high-resolution image
     */
    arma::mat Generate(const arma::mat& lrImage);

    /**
     * @brief Save model weights
     * @param generatorPath Generator model file path
     * @param discriminatorPath Discriminator model file path
     */
    void SaveModel(const std::string& generatorPath,
                   const std::string& discriminatorPath);

    /**
     * @brief Load model weights
     * @param generatorPath Generator model file path
     * @param discriminatorPath Discriminator model file path
     */
    void LoadModel(const std::string& generatorPath,
                   const std::string& discriminatorPath);

private:
    // Network components
    std::unique_ptr<FFN<>> generator;
    std::unique_ptr<FFN<>> discriminator;
    std::unique_ptr<FFN<>> featureExtractor; // VGG for perceptual loss

    // Optimizers
    std::unique_ptr<Adam> generatorOptimizer;
    std::unique_ptr<Adam> discriminatorOptimizer;

    // Model parameters
    int scaleFactor;
    int imgChannels;
    std::string attentionType;
    int lrPatchSize;
    int hrPatchSize;

    // Loss weights
    double adversarialWeight = 1e-3;
    double perceptualWeight = 1.0;
    double pixelWeight = 1.0;

    /**
     * @brief Build the generator network with attention mechanisms
     */
    void BuildGenerator();

    /**
     * @brief Build the discriminator network
     */
    void BuildDiscriminator();

    /**
     * @brief Build feature extractor for perceptual loss (VGG-based)
     */
    void BuildFeatureExtractor();

    /**
     * @brief Channel Attention Module
     */
    void AddChannelAttention(FFN<>& model, int channels);

    /**
     * @brief Spatial Attention Module
     */
    void AddSpatialAttention(FFN<>& model, int channels);

    /**
     * @brief Residual Channel Attention Block (RCAB)
     */
    void AddRCABlock(FFN<>& model, int channels, int reduction = 16);

    /**
     * @brief Residual-in-Residual Dense Block (RRDB)
     */
    void AddRRDBlock(FFN<>& model, int channels);

    /**
     * @brief Pixel shuffle layer for upsampling
     */
    void AddPixelShuffle(FFN<>& model, int channels, int scale);

    /**
     * @brief Calculate generator loss
     */
    double CalculateGeneratorLoss(const arma::mat& generated,
                                  const arma::mat& real,
                                  const arma::mat& featuresGenerated,
                                  const arma::mat& featuresReal);

    /**
     * @brief Calculate discriminator loss
     */
    double CalculateDiscriminatorLoss(const arma::mat& predictionsReal,
                                      const arma::mat& predictionsFake);

    /**
     * @brief Extract image patches for training
     */
    std::vector<arma::mat> ExtractPatches(const arma::mat& images, 
                                         int patchSize, 
                                         int numPatches);

    /**
     * @brief Initialize weights using He initialization
     */
    void InitializeWeights(FFN<>& model);
};

#endif