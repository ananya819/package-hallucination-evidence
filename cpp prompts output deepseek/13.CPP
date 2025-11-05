#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/margin_loss.hpp>
#include <mlpack/methods/preprocess/scaler.hpp>
#include <cmath>
#include <memory>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * Squash activation function for capsules
 * Ensures vector length between 0 and 1 while preserving direction
 */
template<typename MatType = arma::mat>
class SquashFunction : public Layer<MatType>
{
public:
    SquashFunction() : Layer<MatType>() {}

    void Forward(const MatType& input, MatType& output) override
    {
        output.set_size(input.n_rows, input.n_cols);
        
        for (size_t i = 0; i < input.n_cols; ++i)
        {
            MatType capsule = input.col(i);
            double squaredNorm = arma::dot(capsule, capsule);
            double norm = std::sqrt(squaredNorm);
            
            // Squash function: v_j = (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
            double scale = squaredNorm / (1.0 + squaredNorm);
            output.col(i) = (scale / norm) * capsule;
        }
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        g.set_size(input.n_rows, input.n_cols);
        
        for (size_t i = 0; i < input.n_cols; ++i)
        {
            MatType capsule = input.col(i);
            double squaredNorm = arma::dot(capsule, capsule);
            double norm = std::sqrt(squaredNorm);
            
            // Derivative of squash function
            double scale = squaredNorm / (1.0 + squaredNorm);
            MatType jacobian = (scale / norm) * arma::eye<MatType>(input.n_rows, input.n_rows) +
                             (scale / (norm * norm * norm)) * capsule * capsule.t();
            
            g.col(i) = jacobian * gy.col(i);
        }
    }

    virtual SquashFunction* Clone() const
    {
        return new SquashFunction(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
    }
};

/**
 * Primary Capsule Layer
 * Converts CNN features to initial capsules
 */
template<typename MatType = arma::mat>
class PrimaryCapsule : public Layer<MatType>
{
public:
    PrimaryCapsule(const size_t inChannels,
                   const size_t outCapsules,
                   const size_t capsDim,
                   const size_t kernelSize = 9,
                   const size_t stride = 2) :
        Layer<MatType>(),
        inChannels(inChannels),
        outCapsules(outCapsules),
        capsDim(capsDim),
        kernelSize(kernelSize),
        stride(stride)
    {
        // Initialize convolutional weights for capsule generation
        weights.set_size(kernelSize * kernelSize * inChannels, outCapsules * capsDim);
        RandomInitialization().Initialize(weights, 
                                        kernelSize * kernelSize * inChannels, 
                                        outCapsules * capsDim);
        weights *= 0.01; // Small initial weights
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input shape: (height * width * channels, batch_size)
        // For MNIST: 20x20x32 after first conv layer
        const size_t batchSize = input.n_cols;
        const size_t spatialSize = input.n_rows / inChannels;
        const size_t spatialDim = std::sqrt(spatialSize);
        
        // Output: (capsDim * outCapsules, batch_size)
        output.set_size(capsDim * outCapsules, batchSize);
        output.zeros();
        
        // Apply convolutional operation to generate capsules
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType inputImage = input.col(b);
            inputImage.reshape(spatialDim, spatialDim * inChannels);
            
            for (size_t cap = 0; cap < outCapsules; ++cap)
            {
                // Extract weights for this capsule
                MatType capWeights = weights.cols(cap * capsDim, (cap + 1) * capsDim - 1);
                
                // Apply convolution (simplified - using matrix multiplication)
                MatType capOutput = capWeights.t() * inputImage;
                
                // Store in output
                output.rows(cap * capsDim, (cap + 1) * capsDim - 1, b) = 
                    arma::vectorise(capOutput);
            }
        }
        
        // Apply squash activation
        squash.Forward(output, output);
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // Backward through squash
        MatType squashGrad;
        squash.Backward(input, gy, squashGrad);
        
        // Simplified backward pass for convolution
        g = squashGrad;
    }

    virtual PrimaryCapsule* Clone() const
    {
        return new PrimaryCapsule(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inChannels));
        ar(CEREAL_NVP(outCapsules));
        ar(CEREAL_NVP(capsDim));
        ar(CEREAL_NVP(kernelSize));
        ar(CEREAL_NVP(stride));
        ar(CEREAL_NVP(weights));
    }

private:
    size_t inChannels;
    size_t outCapsules;
    size_t capsDim;
    size_t kernelSize;
    size_t stride;
    MatType weights;
    SquashFunction<MatType> squash;
};

/**
 * Dynamic Routing between Capsule Layers
 * Implements the routing-by-agreement mechanism
 */
template<typename MatType = arma::mat>
class CapsuleRouting : public Layer<MatType>
{
public:
    CapsuleRouting(const size_t inCapsules,
                   const size_t inCapsDim,
                   const size_t outCapsules,
                   const size_t outCapsDim,
                   const size_t numRouting = 3) :
        Layer<MatType>(),
        inCapsules(inCapsules),
        inCapsDim(inCapsDim),
        outCapsules(outCapsules),
        outCapsDim(outCapsDim),
        numRouting(numRouting)
    {
        InitializeRoutingWeights();
    }

    void InitializeRoutingWeights()
    {
        // Weight matrix for transforming input capsules to output capsules
        routingWeights.set_size(inCapsDim * outCapsules, outCapsDim * inCapsules);
        RandomInitialization().Initialize(routingWeights, 
                                        inCapsDim * outCapsules, 
                                        outCapsDim * inCapsules);
        routingWeights *= 0.01;
        
        // Initialize coupling coefficients (logits)
        couplingCoefficients.set_size(outCapsules, inCapsules);
        couplingCoefficients.zeros();
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input shape: (inCapsDim * inCapsules, batch_size)
        const size_t batchSize = input.n_cols;
        
        output.set_size(outCapsDim * outCapsules, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType inputCapsules = input.col(b);
            
            // Dynamic routing algorithm
            MatType predictions = ComputePredictions(inputCapsules);
            MatType routedCapsules = RouteCapsules(predictions, b);
            
            output.col(b) = routedCapsules;
        }
    }

    MatType ComputePredictions(const MatType& inputCapsules)
    {
        // Reshape input capsules
        inputCapsules.reshape(inCapsDim, inCapsules);
        
        // Compute prediction vectors u_hat = W * u
        MatType predictions(outCapsDim * outCapsules, inCapsules);
        
        for (size_t i = 0; i < inCapsules; ++i)
        {
            MatType inputCapsule = inputCapsules.col(i);
            
            for (size_t j = 0; j < outCapsules; ++j)
            {
                MatType weightBlock = routingWeights.submat(j * inCapsDim, i * outCapsDim,
                                                          (j + 1) * inCapsDim - 1, 
                                                          (i + 1) * outCapsDim - 1);
                
                MatType prediction = weightBlock.t() * inputCapsule;
                predictions.rows(j * outCapsDim, (j + 1) * outCapsDim - 1, i) = prediction;
            }
        }
        
        return predictions;
    }

    MatType RouteCapsules(const MatType& predictions, size_t batchIdx)
    {
        // Initialize coupling coefficients for this batch
        MatType b = arma::zeros<MatType>(outCapsules, inCapsules);
        
        // Routing iterations
        for (size_t iter = 0; iter < numRouting; ++iter)
        {
            // Compute coupling coefficients using softmax
            MatType c = arma::exp(b);
            c = c / arma::repmat(arma::sum(c, 0), outCapsules, 1);
            
            // Compute weighted sum of predictions
            MatType s = arma::zeros<MatType>(outCapsDim, outCapsules);
            
            for (size_t j = 0; j < outCapsules; ++j)
            {
                for (size_t i = 0; i < inCapsules; ++i)
                {
                    MatType prediction = predictions.rows(j * outCapsDim, 
                                                        (j + 1) * outCapsDim - 1, i);
                    s.col(j) += c(j, i) * prediction;
                }
            }
            
            // Apply squash activation
            MatType v = arma::zeros<MatType>(outCapsDim, outCapsules);
            for (size_t j = 0; j < outCapsules; ++j)
            {
                MatType capsule = s.col(j);
                double squaredNorm = arma::dot(capsule, capsule);
                double norm = std::sqrt(squaredNorm);
                double scale = squaredNorm / (1.0 + squaredNorm);
                v.col(j) = (scale / norm) * capsule;
            }
            
            // Update agreement (only if not last iteration)
            if (iter < numRouting - 1)
            {
                for (size_t j = 0; j < outCapsules; ++j)
                {
                    for (size_t i = 0; i < inCapsules; ++i)
                    {
                        MatType prediction = predictions.rows(j * outCapsDim, 
                                                            (j + 1) * outCapsDim - 1, i);
                        b(j, i) += arma::dot(v.col(j), prediction);
                    }
                }
            }
            else
            {
                // Return final output capsules
                return arma::vectorise(v);
            }
        }
        
        return arma::zeros<MatType>(outCapsDim * outCapsules, 1);
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // Simplified backward pass
        // In practice, this would compute gradients through the routing procedure
        g.set_size(input.n_rows, input.n_cols);
        g = gy; // Placeholder
    }

    virtual CapsuleRouting* Clone() const
    {
        return new CapsuleRouting(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inCapsules));
        ar(CEREAL_NVP(inCapsDim));
        ar(CEREAL_NVP(outCapsules));
        ar(CEREAL_NVP(outCapsDim));
        ar(CEREAL_NVP(numRouting));
        ar(CEREAL_NVP(routingWeights));
        ar(CEREAL_NVP(couplingCoefficients));
    }

private:
    size_t inCapsules;
    size_t inCapsDim;
    size_t outCapsules;
    size_t outCapsDim;
    size_t numRouting;
    
    MatType routingWeights;
    MatType couplingCoefficients;
};

/**
 * Margin Loss for Capsule Networks
 * Specialized loss function that maximizes presence of correct class capsule
 */
class CapsuleMarginLoss
{
public:
    CapsuleMarginLoss(const double mPlus = 0.9,
                      const double mMinus = 0.1,
                      const double lambda = 0.5) :
        mPlus(mPlus),
        mMinus(mMinus),
        lambda(lambda)
    {}

    template<typename PredictionType, typename TargetType>
    static double Forward(const PredictionType& prediction,
                         const TargetType& target)
    {
        double loss = 0.0;
        const size_t numClasses = prediction.n_rows;
        const size_t batchSize = prediction.n_cols;
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            // Get capsule norms (lengths) as class probabilities
            vec capsuleNorms(numClasses);
            for (size_t c = 0; c < numClasses; ++c)
            {
                // Assuming each class has 16D capsule
                const size_t capsDim = 16;
                mat classCapsule = prediction.submat(c * capsDim, b, 
                                                   (c + 1) * capsDim - 1, b);
                capsuleNorms(c) = arma::norm(classCapsule);
            }
            
            // Find true class
            uword trueClass;
            target.col(b).max(trueClass);
            
            // Margin loss calculation
            for (size_t c = 0; c < numClasses; ++c)
            {
                double T_c = (c == trueClass) ? 1.0 : 0.0;
                double m = (c == trueClass) ? mPlus : mMinus;
                
                loss += T_c * std::max(0.0, m - capsuleNorms(c)) *
                       std::max(0.0, m - capsuleNorms(c)) +
                       lambda * (1 - T_c) * std::max(0.0, capsuleNorms(c) - mMinus) *
                       std::max(0.0, capsuleNorms(c) - mMinus);
            }
        }
        
        return loss / batchSize;
    }

    template<typename PredictionType, typename TargetType, typename LossType>
    static void Backward(const PredictionType& prediction,
                        const TargetType& target,
                        LossType& loss)
    {
        loss.zeros();
        const size_t numClasses = prediction.n_rows / 16; // Assuming 16D capsules
        const size_t batchSize = prediction.n_cols;
        const size_t capsDim = 16;
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            // Compute capsule norms
            vec capsuleNorms(numClasses);
            for (size_t c = 0; c < numClasses; ++c)
            {
                mat classCapsule = prediction.submat(c * capsDim, b, 
                                                   (c + 1) * capsDim - 1, b);
                capsuleNorms(c) = arma::norm(classCapsule);
            }
            
            // Find true class
            uword trueClass;
            target.col(b).max(trueClass);
            
            // Compute gradients
            for (size_t c = 0; c < numClasses; ++c)
            {
                double T_c = (c == trueClass) ? 1.0 : 0.0;
                double m = (c == trueClass) ? mPlus : mMinus;
                double capsuleNorm = capsuleNorms(c);
                
                double gradientScale = 0.0;
                if (T_c == 1.0)
                {
                    gradientScale = -2.0 * T_c * (m - capsuleNorm) * 
                                  (capsuleNorm > 0 ? 1.0 : 0.0);
                }
                else
                {
                    gradientScale = 2.0 * lambda * (1 - T_c) * 
                                  (capsuleNorm - mMinus) * 
                                  (capsuleNorm > mMinus ? 1.0 : 0.0);
                }
                
                // Apply gradient to capsule vector
                if (capsuleNorm > 0)
                {
                    mat classCapsule = prediction.submat(c * capsDim, b, 
                                                       (c + 1) * capsDim - 1, b);
                    loss.submat(c * capsDim, b, (c + 1) * capsDim - 1, b) = 
                        gradientScale * (classCapsule / capsuleNorm);
                }
            }
        }
        
        loss = loss / batchSize;
    }

private:
    double mPlus;
    double mMinus;
    double lambda;
};

/**
 * Complete Capsule Network for MNIST Classification
 */
class CapsuleNetwork
{
public:
    CapsuleNetwork(const size_t imageWidth = 28,
                   const size_t imageHeight = 28,
                   const size_t numClasses = 10) :
        imageWidth(imageWidth),
        imageHeight(imageHeight),
        numClasses(numClasses)
    {
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        network = std::make_unique<FFN<CapsuleMarginLoss, RandomInitialization>>();
        
        std::cout << "Building Capsule Network for MNIST..." << std::endl;
        
        // Initial convolutional layer
        network->Add<Convolution<>>(1, 256, 9, 9, 1, 1, 0, 0); // 28x28x1 -> 20x20x256
        network->Add<ReLULayer<>>();
        
        std::cout << "Conv1: 1->256 channels, 9x9 kernel" << std::endl;
        
        // Primary capsule layer
        // 20x20x256 -> 6x6x32 capsules of 8D
        network->Add<PrimaryCapsule<>>(256, 32, 8, 9, 2);
        std::cout << "Primary Caps: 32 capsules, 8D each" << std::endl;
        
        // Digit capsule layer with dynamic routing
        // 32x8D capsules -> 10x16D capsules
        network->Add<CapsuleRouting<>>(32, 8, numClasses, 16, 3);
        std::cout << "Digit Caps: 10 capsules, 16D each" << std::endl;
        
        std::cout << "Network architecture complete." << std::endl;
    }

    /**
     * Train the capsule network
     */
    void Train(const mat& trainImages, const mat& trainLabels,
               const mat& testImages = mat(), const mat& testLabels = mat(),
               const size_t epochs = 50, const size_t batchSize = 32)
    {
        std::cout << "\nTraining Capsule Network..." << std::endl;
        std::cout << "Training samples: " << trainImages.n_cols << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        
        ens::Adam optimizer(0.001, batchSize, 0.9, 0.999, 1e-8);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            double epochLoss = 0.0;
            size_t numBatches = 0;
            
            for (size_t i = 0; i < trainImages.n_cols; i += batchSize)
            {
                size_t currentBatchSize = std::min(batchSize, trainImages.n_cols - i);
                
                mat batchImages = trainImages.cols(i, i + currentBatchSize - 1);
                mat batchLabels = trainLabels.cols(i, i + currentBatchSize - 1);
                
                double loss = network->Evaluate(batchImages, batchLabels);
                network->Gradient(batchImages, batchLabels);
                optimizer.Update(network->Parameters().parameters(),
                               network->Parameters().gradient());
                
                epochLoss += loss;
                numBatches++;
                
                if (numBatches % 100 == 0)
                {
                    std::cout << "Epoch " << epoch + 1 << ", Batch " << numBatches 
                              << ", Loss: " << loss << std::endl;
                }
            }
            
            double avgLoss = epochLoss / numBatches;
            
            if (testImages.n_cols > 0)
            {
                double accuracy = Evaluate(testImages, testLabels);
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                          << " - Loss: " << avgLoss
                          << " - Test Accuracy: " << accuracy << "%" << std::endl;
            }
            else
            {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                          << " - Loss: " << avgLoss << std::endl;
            }
        }
    }

    /**
     * Evaluate model accuracy
     */
    double Evaluate(const mat& testImages, const mat& testLabels)
    {
        size_t correct = 0;
        size_t total = testImages.n_cols;
        
        for (size_t i = 0; i < total; ++i)
        {
            mat prediction;
            network->Predict(testImages.col(i), prediction);
            
            // Find class with maximum capsule norm
            uword predictedClass = PredictClassFromCapsules(prediction);
            uword trueClass;
            testLabels.col(i).max(trueClass);
            
            if (predictedClass == trueClass)
                correct++;
        }
        
        return (static_cast<double>(correct) / total) * 100.0;
    }

    /**
     * Predict class from capsule outputs
     */
    uword PredictClassFromCapsules(const mat& capsules)
    {
        // Capsules are 16D vectors for each class
        const size_t capsDim = 16;
        const size_t numCaps = capsules.n_rows / capsDim;
        
        vec capsuleNorms(numCaps);
        for (size_t i = 0; i < numCaps; ++i)
        {
            mat capsule = capsules.rows(i * capsDim, (i + 1) * capsDim - 1);
            capsuleNorms(i) = arma::norm(capsule);
        }
        
        uword predictedClass;
        capsuleNorms.max(predictedClass);
        return predictedClass;
    }

    /**
     * Get capsule activations for visualization
     */
    mat GetCapsuleActivations(const mat& image)
    {
        mat activations;
        network->Predict(image, activations);
        return activations;
    }

    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "capsule_network", *network);
        std::cout << "Capsule Network saved to: " << filename << std::endl;
    }

    void LoadModel(const std::string& filename)
    {
        data::Load(filename, "capsule_network", *network);
        std::cout << "Capsule Network loaded from: " << filename << std::endl;
    }

private:
    size_t imageWidth;
    size_t imageHeight;
    size_t numClasses;
    
    std::unique_ptr<FFN<CapsuleMarginLoss, RandomInitialization>> network;
};

/**
 * MNIST Data Loader and Preprocessor for Capsule Networks
 */
class MNISTDataLoader
{
public:
    MNISTDataLoader(const std::string& dataPath = "data/mnist") :
        dataPath(dataPath)
    {}

    void LoadData(mat& trainImages, mat& trainLabels,
                  mat& testImages, mat& testLabels,
                  size_t maxSamples = 0)
    {
        std::cout << "Loading MNIST data..." << std::endl;
        
        // In practice, you would load actual MNIST files here
        // For this example, we'll generate synthetic data
        
        const size_t totalTrainSamples = maxSamples > 0 ? 
            std::min(maxSamples, size_t(60000)) : 60000;
        const size_t totalTestSamples = maxSamples > 0 ? 
            std::min(maxSamples, size_t(10000)) : 10000;
        
        // Generate synthetic MNIST-like data
        GenerateSyntheticMNIST(trainImages, trainLabels, totalTrainSamples);
        GenerateSyntheticMNIST(testImages, testLabels, totalTestSamples);
        
        std::cout << "Loaded " << trainImages.n_cols << " training samples" << std::endl;
        std::cout << "Loaded " << testImages.n_cols << " test samples" << std::endl;
    }

private:
    std::string dataPath;

    void GenerateSyntheticMNIST(mat& images, mat& labels, size_t numSamples)
    {
        images.set_size(28 * 28, numSamples);
        labels.set_size(10, numSamples);
        labels.zeros();
        
        for (size_t i = 0; i < numSamples; ++i)
        {
            // Generate random digit-like patterns
            size_t digit = i % 10;
            mat digitImage = arma::randu<mat>(28, 28) * 0.3;
            
            // Add some structure to make it look like a digit
            AddDigitStructure(digitImage, digit);
            
            images.col(i) = arma::vectorise(digitImage);
            labels(digit, i) = 1.0;
        }
    }

    void AddDigitStructure(mat& image, size_t digit)
    {
        // Add simple geometric patterns to simulate digits
        switch (digit)
        {
            case 0: // Circle
                AddCircle(image, 14, 14, 8);
                break;
            case 1: // Vertical line
                AddLine(image, 14, 5, 14, 23);
                break;
            case 2: // Curve and line
                AddArc(image, 14, 14, 8, 0, M_PI);
                AddLine(image, 6, 15, 22, 15);
                break;
            // Add more cases for other digits...
            default:
                AddRectangle(image, 8, 8, 20, 20);
                break;
        }
    }

    void AddCircle(mat& image, double cx, double cy, double radius)
    {
        for (size_t i = 0; i < 28; ++i)
        {
            for (size_t j = 0; j < 28; ++j)
            {
                double dx = i - cx;
                double dy = j - cy;
                double dist = std::sqrt(dx * dx + dy * dy);
                if (std::abs(dist - radius) < 2.0)
                {
                    image(i, j) = 1.0;
                }
            }
        }
    }

    void AddLine(mat& image, double x1, double y1, double x2, double y2)
    {
        // Simplified line drawing
        double steps = std::max(std::abs(x2 - x1), std::abs(y2 - y1));
        for (double t = 0; t <= 1.0; t += 1.0 / steps)
        {
            int x = x1 + t * (x2 - x1);
            int y = y1 + t * (y2 - y1);
            if (x >= 0 && x < 28 && y >= 0 && y < 28)
            {
                image(x, y) = 1.0;
            }
        }
    }

    void AddArc(mat& image, double cx, double cy, double radius, 
                double startAngle, double endAngle)
    {
        for (double angle = startAngle; angle <= endAngle; angle += 0.1)
        {
            int x = cx + radius * std::cos(angle);
            int y = cy + radius * std::sin(angle);
            if (x >= 0 && x < 28 && y >= 0 && y < 28)
            {
                image(x, y) = 1.0;
            }
        }
    }

    void AddRectangle(mat& image, double x1, double y1, double x2, double y2)
    {
        for (int i = x1; i <= x2; ++i)
        {
            for (int j = y1; j <= y2; ++j)
            {
                if (i >= 0 && i < 28 && j >= 0 && j < 28)
                {
                    image(i, j) = 1.0;
                }
            }
        }
    }
};

/**
 * Main example: MNIST classification with Capsule Network
 */
int main()
{
    std::cout << "Capsule Network with Dynamic Routing for MNIST" << std::endl;
    std::cout << "==============================================" << std::endl;
    
    // Load MNIST data
    MNISTDataLoader dataLoader;
    mat trainImages, trainLabels, testImages, testLabels;
    
    dataLoader.LoadData(trainImages, trainLabels, testImages, testLabels, 1000);
    
    // Create capsule network
    CapsuleNetwork capsuleNet(28, 28, 10);
    
    // Train the network
    capsuleNet.Train(trainImages, trainLabels, testImages, testLabels, 10, 32);
    
    // Evaluate final performance
    double finalAccuracy = capsuleNet.Evaluate(testImages, testLabels);
    std::cout << "\nFinal Test Accuracy: " << finalAccuracy << "%" << std::endl;
    
    // Save the trained model
    capsuleNet.SaveModel("capsule_network_mnist.xml");
    
    // Test prediction on a single image
    mat testImage = testImages.col(0);
    mat capsuleActivations = capsuleNet.GetCapsuleActivations(testImage);
    uword predictedClass = capsuleNet.PredictClassFromCapsules(capsuleActivations);
    
    std::cout << "Predicted class for first test image: " << predictedClass << std::endl;
    
    return 0;
}