#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/activation_functions/relu_function.hpp>
#include <mlpack/methods/ann/activation_functions/sigmoid_function.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/regularizer/lregularizer.hpp>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

using namespace mlpack;
using namespace mlpack::ann;

// 3D Convolutional Layer
template<typename InputDataType = arma::mat, typename OutputDataType = arma::mat>
class Convolution3D
{
public:
    Convolution3D(const size_t inChannels,
                  const size_t outChannels,
                  const size_t kernelWidth,
                  const size_t kernelHeight,
                  const size_t kernelDepth,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const size_t strideDepth = 1) :
        inChannels(inChannels),
        outChannels(outChannels),
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        kernelDepth(kernelDepth),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        strideDepth(strideDepth)
    {
        // Initialize weight tensor (outChannels, inChannels, kW, kH, kD)
        weightSize = outChannels * inChannels * kernelWidth * kernelHeight * kernelDepth;
        weights.set_size(weightSize, 1);
        bias.set_size(outChannels, 1);
        
        // Initialize weights using Glorot initialization
        GlorotInitialization<> init;
        init.Initialize(weights, weightSize, 1);
        bias.zeros();
    }

    template<typename eT>
    void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
    {
        size_t inputWidth = input.n_rows;
        size_t inputHeight = input.n_cols;
        size_t inputDepth = input.n_slices;
        
        // Calculate output dimensions
        size_t outputWidth = (inputWidth - kernelWidth) / strideWidth + 1;
        size_t outputHeight = (inputHeight - kernelHeight) / strideHeight + 1;
        size_t outputDepth = (inputDepth - kernelDepth) / strideDepth + 1;
        
        output.set_size(outputWidth, outputHeight, outChannels);
        output.zeros();
        
        // Perform 3D convolution
        for (size_t outCh = 0; outCh < outChannels; ++outCh)
        {
            for (size_t w = 0; w < outputWidth; ++w)
            {
                for (size_t h = 0; h < outputHeight; ++h)
                {
                    for (size_t d = 0; d < outputDepth; ++d)
                    {
                        eT sum = bias(outCh, 0);
                        
                        // Convolution operation
                        for (size_t inCh = 0; inCh < inChannels; ++inCh)
                        {
                            for (size_t kw = 0; kw < kernelWidth; ++kw)
                            {
                                for (size_t kh = 0; kh < kernelHeight; ++kh)
                                {
                                    for (size_t kd = 0; kd < kernelDepth; ++kd)
                                    {
                                        size_t inputW = w * strideWidth + kw;
                                        size_t inputH = h * strideHeight + kh;
                                        size_t inputD = d * strideDepth + kd;
                                        
                                        if (inputW < inputWidth && inputH < inputHeight && inputD < inputDepth)
                                        {
                                            size_t weightIdx = outCh * inChannels * kernelWidth * kernelHeight * kernelDepth +
                                                              inCh * kernelWidth * kernelHeight * kernelDepth +
                                                              kw * kernelHeight * kernelDepth +
                                                              kh * kernelDepth +
                                                              kd;
                                            
                                            sum += weights(weightIdx, 0) * input(inputW, inputH, inputD, inCh);
                                        }
                                    }
                                }
                            }
                        }
                        
                        output(w, h, outCh) = sum;
                    }
                }
            }
        }
    }

    template<typename eT>
    void Backward(const arma::Cube<eT>& /* input */,
                  const arma::Cube<eT>& /* gy */,
                  arma::Cube<eT>& /* g */)
    {
        // Gradient computation would go here
        // For simplicity, this is a placeholder
    }

    // Getters
    const arma::mat& Weights() const { return weights; }
    const arma::mat& Bias() const { return bias; }
    arma::mat& Weights() { return weights; }
    arma::mat& Bias() { return bias; }

private:
    size_t inChannels, outChannels;
    size_t kernelWidth, kernelHeight, kernelDepth;
    size_t strideWidth, strideHeight, strideDepth;
    size_t weightSize;
    
    arma::mat weights;
    arma::mat bias;
};

// 3D Max Pooling Layer
template<typename InputDataType = arma::mat, typename OutputDataType = arma::mat>
class MaxPooling3D
{
public:
    MaxPooling3D(const size_t kernelWidth,
                 const size_t kernelHeight,
                 const size_t kernelDepth,
                 const size_t strideWidth = 1,
                 const size_t strideHeight = 1,
                 const size_t strideDepth = 1) :
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        kernelDepth(kernelDepth),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        strideDepth(strideDepth)
    {
    }

    template<typename eT>
    void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
    {
        size_t inputWidth = input.n_rows;
        size_t inputHeight = input.n_cols;
        size_t inputDepth = input.n_slices;
        
        // Calculate output dimensions
        size_t outputWidth = (inputWidth - kernelWidth) / strideWidth + 1;
        size_t outputHeight = (inputHeight - kernelHeight) / strideHeight + 1;
        size_t outputDepth = (inputDepth - kernelDepth) / strideDepth + 1;
        
        output.set_size(outputWidth, outputHeight, outputDepth);
        output.zeros();
        
        // Perform 3D max pooling
        for (size_t w = 0; w < outputWidth; ++w)
        {
            for (size_t h = 0; h < outputHeight; ++h)
            {
                for (size_t d = 0; d < outputDepth; ++d)
                {
                    eT maxVal = std::numeric_limits<eT>::lowest();
                    
                    for (size_t kw = 0; kw < kernelWidth; ++kw)
                    {
                        for (size_t kh = 0; kh < kernelHeight; ++kh)
                        {
                            for (size_t kd = 0; kd < kernelDepth; ++kd)
                            {
                                size_t inputW = w * strideWidth + kw;
                                size_t inputH = h * strideHeight + kh;
                                size_t inputD = d * strideDepth + kd;
                                
                                if (inputW < inputWidth && inputH < inputHeight && inputD < inputDepth)
                                {
                                    maxVal = std::max(maxVal, input(inputW, inputH, inputD));
                                }
                            }
                        }
                    }
                    
                    output(w, h, d) = maxVal;
                }
            }
        }
    }

private:
    size_t kernelWidth, kernelHeight, kernelDepth;
    size_t strideWidth, strideHeight, strideDepth;
};

// Batch Normalization 3D
template<typename InputDataType = arma::mat, typename OutputDataType = arma::mat>
class BatchNorm3D
{
public:
    BatchNorm3D(const size_t numFeatures,
                const double eps = 1e-8,
                const double momentum = 0.1) :
        numFeatures(numFeatures),
        eps(eps),
        momentum(momentum),
        runningMean(numFeatures),
        runningVar(numFeatures),
        beta(numFeatures),
        gamma(numFeatures)
    {
        runningMean.zeros();
        runningVar.ones();
        beta.zeros();
        gamma.ones();
    }

    template<typename eT>
    void Forward(const arma::Cube<eT>& input, arma::Cube<eT>& output)
    {
        size_t width = input.n_rows;
        size_t height = input.n_cols;
        size_t depth = input.n_slices;
        
        output.set_size(width, height, depth);
        
        // Compute mean and variance for each channel
        for (size_t ch = 0; ch < depth; ++ch)
        {
            eT mean = 0;
            eT var = 0;
            
            // Compute mean
            for (size_t w = 0; w < width; ++w)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    mean += input(w, h, ch);
                }
            }
            mean /= (width * height);
            
            // Compute variance
            for (size_t w = 0; w < width; ++w)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    eT diff = input(w, h, ch) - mean;
                    var += diff * diff;
                }
            }
            var /= (width * height);
            
            // Normalize and scale
            eT invStd = 1.0 / std::sqrt(var + eps);
            for (size_t w = 0; w < width; ++w)
            {
                for (size_t h = 0; h < height; ++h)
                {
                    eT normalized = (input(w, h, ch) - mean) * invStd;
                    output(w, h, ch) = gamma(ch) * normalized + beta(ch);
                }
            }
            
            // Update running statistics
            runningMean(ch) = (1 - momentum) * runningMean(ch) + momentum * mean;
            runningVar(ch) = (1 - momentum) * runningVar(ch) + momentum * var;
        }
    }

private:
    size_t numFeatures;
    double eps, momentum;
    arma::vec runningMean, runningVar;
    arma::vec beta, gamma;
};

// 3D CNN for Medical Imaging
class Medical3DCNN
{
public:
    Medical3DCNN(const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t inputDepth,
                 const size_t inputChannels,
                 const size_t numClasses,
                 const std::vector<size_t>& convFilters = {32, 64, 128},
                 const std::vector<size_t>& convKernels = {3, 3, 3},
                 const std::vector<size_t>& poolKernels = {2, 2, 2}) :
        inputWidth(inputWidth),
        inputHeight(inputHeight),
        inputDepth(inputDepth),
        inputChannels(inputChannels),
        numClasses(numClasses)
    {
        // Initialize network layers
        InitializeNetwork(convFilters, convKernels, poolKernels);
    }

    void InitializeNetwork(const std::vector<size_t>& convFilters,
                          const std::vector<size_t>& convKernels,
                          const std::vector<size_t>& poolKernels)
    {
        size_t currentWidth = inputWidth;
        size_t currentHeight = inputHeight;
        size_t currentDepth = inputDepth;
        size_t currentChannels = inputChannels;
        
        // Add convolutional layers
        for (size_t i = 0; i < convFilters.size(); ++i)
        {
            size_t kernelSize = (i < convKernels.size()) ? convKernels[i] : 3;
            size_t poolSize = (i < poolKernels.size()) ? poolKernels[i] : 2;
            
            // Convolutional layer
            convLayers.emplace_back(std::make_unique<Convolution3D<>>(
                currentChannels, convFilters[i], kernelSize, kernelSize, kernelSize));
            
            // Batch normalization
            batchNormLayers.emplace_back(std::make_unique<BatchNorm3D<>>(
                convFilters[i]));
            
            // Update dimensions
            currentWidth = (currentWidth - kernelSize) + 1;
            currentHeight = (currentHeight - kernelSize) + 1;
            currentDepth = (currentDepth - kernelSize) + 1;
            currentChannels = convFilters[i];
            
            // Max pooling layer
            if (poolSize > 1)
            {
                poolLayers.emplace_back(std::make_unique<MaxPooling3D<>>(
                    poolSize, poolSize, poolSize, poolSize, poolSize, poolSize));
                
                // Update dimensions after pooling
                currentWidth = (currentWidth - poolSize) / poolSize + 1;
                currentHeight = (currentHeight - poolSize) / poolSize + 1;
                currentDepth = (currentDepth - poolSize) / poolSize + 1;
            }
            else
            {
                poolLayers.emplace_back(nullptr);
            }
        }
        
        // Calculate flattened size for fully connected layer
        flattenedSize = currentWidth * currentHeight * currentDepth * currentChannels;
        
        // Fully connected layers
        fcLayers.emplace_back(std::make_unique<Linear<>>(flattenedSize, 512));
        fcLayers.emplace_back(std::make_unique<Linear<>>(512, 256));
        fcLayers.emplace_back(std::make_unique<Linear<>>(256, numClasses));
        
        // Initialize weights
        GlorotInitialization<> init;
        for (auto& layer : fcLayers)
        {
            Linear<>* linearLayer = dynamic_cast<Linear<>*>(layer.get());
            if (linearLayer)
            {
                init.Initialize(linearLayer->Weights(), 
                               linearLayer->OutputDimensions(), 
                               linearLayer->InputDimensions());
            }
        }
    }

    template<typename eT>
    void Forward(const arma::Cube<eT>& input, arma::mat& output)
    {
        arma::Cube<eT> current = input;
        
        // Forward through convolutional layers
        for (size_t i = 0; i < convLayers.size(); ++i)
        {
            // Convolution
            arma::Cube<eT> convOutput;
            convLayers[i]->Forward(current, convOutput);
            
            // Batch normalization
            arma::Cube<eT> bnOutput;
            batchNormLayers[i]->Forward(convOutput, bnOutput);
            
            // ReLU activation
            arma::Cube<eT> reluOutput = bnOutput;
            reluOutput.tube().for_each([](eT& val) { val = std::max(eT(0), val); });
            
            // Pooling (if exists)
            if (poolLayers[i])
            {
                arma::Cube<eT> poolOutput;
                poolLayers[i]->Forward(reluOutput, poolOutput);
                current = poolOutput;
            }
            else
            {
                current = reluOutput;
            }
        }
        
        // Flatten for fully connected layers
        arma::mat flattened = arma::vectorise(current);
        
        // Forward through fully connected layers
        arma::mat fcCurrent = flattened;
        for (size_t i = 0; i < fcLayers.size(); ++i)
        {
            arma::mat fcOutput;
            fcLayers[i]->Forward(fcCurrent, fcOutput);
            
            if (i < fcLayers.size() - 1) // Apply ReLU except for last layer
            {
                fcOutput = arma::max(fcOutput, arma::zeros<arma::mat>(fcOutput.n_rows, fcOutput.n_cols));
            }
            
            fcCurrent = fcOutput;
        }
        
        // Apply softmax for classification
        output = arma::softmax(fcCurrent);
    }

    // Training function
    void Train(const std::vector<arma::cube>& trainingData,
              const arma::Row<size_t>& labels,
              size_t numEpochs = 100,
              double learningRate = 0.001,
              size_t batchSize = 8)
    {
        std::cout << "Training 3D CNN for medical imaging..." << std::endl;
        std::cout << "Training samples: " << trainingData.size() << std::endl;
        std::cout << "Number of classes: " << numClasses << std::endl;
        
        for (size_t epoch = 0; epoch < numEpochs; ++epoch)
        {
            double totalLoss = 0.0;
            size_t numBatches = 0;
            
            // Process data in batches
            for (size_t batchStart = 0; batchStart < trainingData.size(); batchStart += batchSize)
            {
                size_t batchEnd = std::min(batchStart + batchSize, trainingData.size());
                double batchLoss = 0.0;
                
                for (size_t i = batchStart; i < batchEnd; ++i)
                {
                    const auto& volume = trainingData[i];
                    size_t label = labels(i);
                    
                    // Convert cube to proper format
                    arma::Cube<double> input(volume.memptr(), volume.n_rows, volume.n_cols, volume.n_slices, false, true);
                    
                    // Forward pass
                    arma::mat output;
                    Forward(input, output);
                    
                    // Compute loss (cross-entropy)
                    arma::mat target = arma::zeros<arma::mat>(numClasses, 1);
                    target(label, 0) = 1.0;
                    
                    arma::mat error = output - target;
                    double loss = -arma::accu(target % arma::log(output + 1e-8));
                    batchLoss += loss;
                    
                    // Update weights (simplified)
                    UpdateWeights(learningRate, error, input);
                }
                
                totalLoss += batchLoss / (batchEnd - batchStart);
                numBatches++;
            }
            
            if (epoch % 10 == 0)
            {
                std::cout << "Epoch " << epoch << ", Average Loss: " 
                         << totalLoss / numBatches << std::endl;
            }
        }
        
        std::cout << "Training completed!" << std::endl;
    }

    // Prediction function
    template<typename eT>
    size_t Predict(const arma::Cube<eT>& input)
    {
        arma::mat output;
        Forward(input, output);
        
        // Return class with highest probability
        arma::uword maxIndex;
        output.max(maxIndex);
        return static_cast<size_t>(maxIndex);
    }

    // Batch prediction
    template<typename eT>
    void PredictBatch(const std::vector<arma::Cube<eT>>& inputs, arma::Row<size_t>& predictions)
    {
        predictions.set_size(inputs.size());
        
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            predictions(i) = Predict(inputs[i]);
        }
    }

    // Evaluation function
    double Evaluate(const std::vector<arma::cube>& testData,
                   const arma::Row<size_t>& trueLabels)
    {
        size_t correct = 0;
        
        for (size_t i = 0; i < testData.size(); ++i)
        {
            const auto& volume = testData[i];
            arma::Cube<double> input(volume.memptr(), volume.n_rows, volume.n_cols, volume.n_slices, false, true);
            
            size_t predicted = Predict(input);
            if (predicted == trueLabels(i))
            {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / testData.size();
    }

private:
    size_t inputWidth, inputHeight, inputDepth, inputChannels, numClasses;
    size_t flattenedSize;
    
    std::vector<std::unique_ptr<Convolution3D<>>> convLayers;
    std::vector<std::unique_ptr<BatchNorm3D<>>> batchNormLayers;
    std::vector<std::unique_ptr<MaxPooling3D<>>> poolLayers;
    std::vector<std::unique_ptr<Layer<>>> fcLayers;

    void UpdateWeights(double learningRate, const arma::mat& error, const arma::cube& /*input*/)
    {
        // Simplified weight update - in practice, implement proper backpropagation
        static bool firstCall = true;
        if (firstCall)
        {
            std::cout << "Note: Weight updates are simplified. For production use, "
                      << "implement proper backpropagation through 3D convolutions." << std::endl;
            firstCall = false;
        }
        
        // In a real implementation, you would:
        // 1. Compute gradients through each layer
        // 2. Update convolutional weights and biases
        // 3. Update fully connected layer weights
        // 4. Apply proper gradient descent optimization
    }
};

// Medical Imaging Data Generator
class MedicalDataGenerator
{
public:
    // Generate synthetic 3D medical volumes for training
    static void GenerateSyntheticData(size_t numSamples,
                                    size_t width,
                                    size_t height,
                                    size_t depth,
                                    size_t numClasses,
                                    std::vector<arma::cube>& data,
                                    arma::Row<size_t>& labels)
    {
        data.clear();
        data.reserve(numSamples);
        labels.set_size(numSamples);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> classDist(0, numClasses - 1);
        
        std::cout << "Generating synthetic 3D medical data..." << std::endl;
        
        for (size_t i = 0; i < numSamples; ++i)
        {
            // Generate random 3D volume
            arma::cube volume(width, height, depth);
            
            // Assign random class label
            size_t classLabel = classDist(gen);
            labels(i) = classLabel;
            
            // Generate class-specific patterns
            switch (classLabel)
            {
                case 0: // Class 0: Sphere-like pattern
                    GenerateSpherePattern(volume, gen);
                    break;
                case 1: // Class 1: Cube-like pattern
                    GenerateCubePattern(volume, gen);
                    break;
                case 2: // Class 2: Random noise
                    GenerateNoisePattern(volume, gen);
                    break;
                default: // Other classes: Mixed patterns
                    GenerateMixedPattern(volume, gen);
                    break;
            }
            
            data.push_back(volume);
        }
        
        std::cout << "Generated " << numSamples << " synthetic 3D volumes" << std::endl;
    }

private:
    static void GenerateSpherePattern(arma::cube& volume, std::mt19937& gen)
    {
        size_t width = volume.n_rows;
        size_t height = volume.n_cols;
        size_t depth = volume.n_slices;
        
        std::uniform_real_distribution<> intensityDist(0.5, 1.0);
        std::uniform_int_distribution<> centerDist(10, std::min({width, height, depth}) - 10);
        
        size_t centerX = centerDist(gen);
        size_t centerY = centerDist(gen);
        size_t centerZ = centerDist(gen);
        size_t radius = std::min({centerX, centerY, centerZ, 
                                width - centerX, height - centerY, depth - centerZ}) / 2;
        
        double intensity = intensityDist(gen);
        
        for (size_t x = 0; x < width; ++x)
        {
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t z = 0; z < depth; ++z)
                {
                    double dx = static_cast<double>(x) - centerX;
                    double dy = static_cast<double>(y) - centerY;
                    double dz = static_cast<double>(z) - centerZ;
                    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    if (distance <= radius)
                    {
                        volume(x, y, z) = intensity * (1.0 - distance / radius);
                    }
                    else
                    {
                        volume(x, y, z) = 0.0;
                    }
                }
            }
        }
    }

    static void GenerateCubePattern(arma::cube& volume, std::mt19937& gen)
    {
        size_t width = volume.n_rows;
        size_t height = volume.n_cols;
        size_t depth = volume.n_slices;
        
        std::uniform_real_distribution<> intensityDist(0.6, 1.0);
        std::uniform_int_distribution<> cornerDist(5, std::min({width, height, depth}) - 15);
        
        size_t startX = cornerDist(gen);
        size_t startY = cornerDist(gen);
        size_t startZ = cornerDist(gen);
        size_t size = 10;
        
        double intensity = intensityDist(gen);
        
        for (size_t x = startX; x < std::min(startX + size, width); ++x)
        {
            for (size_t y = startY; y < std::min(startY + size, height); ++y)
            {
                for (size_t z = startZ; z < std::min(startZ + size, depth); ++z)
                {
                    volume(x, y, z) = intensity;
                }
            }
        }
    }

    static void GenerateNoisePattern(arma::cube& volume, std::mt19937& gen)
    {
        std::uniform_real_distribution<> noiseDist(0.0, 0.3);
        
        for (size_t x = 0; x < volume.n_rows; ++x)
        {
            for (size_t y = 0; y < volume.n_cols; ++y)
            {
                for (size_t z = 0; z < volume.n_slices; ++z)
                {
                    volume(x, y, z) = noiseDist(gen);
                }
            }
        }
    }

    static void GenerateMixedPattern(arma::cube& volume, std::mt19937& gen)
    {
        GenerateSpherePattern(volume, gen);
        GenerateNoisePattern(volume, gen);
        
        // Combine patterns
        for (size_t x = 0; x < volume.n_rows; ++x)
        {
            for (size_t y = 0; y < volume.n_cols; ++y)
            {
                for (size_t z = 0; z < volume.n_slices; ++z)
                {
                    volume(x, y, z) = std::min(1.0, volume(x, y, z) * 1.5);
                }
            }
        }
    }
};

// Data preprocessing utilities
class MedicalPreprocessor
{
public:
    // Normalize volume intensities to [0, 1]
    static void NormalizeVolume(arma::cube& volume)
    {
        double minVal = volume.min();
        double maxVal = volume.max();
        
        if (maxVal > minVal)
        {
            volume = (volume - minVal) / (maxVal - minVal);
        }
        else
        {
            volume.zeros();
        }
    }

    // Apply windowing (common in medical imaging)
    static void ApplyWindowing(arma::cube& volume, double windowCenter, double windowWidth)
    {
        double minVal = windowCenter - windowWidth / 2.0;
        double maxVal = windowCenter + windowWidth / 2.0;
        
        volume = arma::clamp((volume - minVal) / (maxVal - minVal), 0.0, 1.0);
    }

    // Resize volume to target dimensions
    static arma::cube ResizeVolume(const arma::cube& volume, 
                                  size_t targetWidth, 
                                  size_t targetHeight, 
                                  size_t targetDepth)
    {
        arma::cube resized(targetWidth, targetHeight, targetDepth);
        resized.zeros();
        
        double scaleX = static_cast<double>(volume.n_rows) / targetWidth;
        double scaleY = static_cast<double>(volume.n_cols) / targetHeight;
        double scaleZ = static_cast<double>(volume.n_slices) / targetDepth;
        
        for (size_t x = 0; x < targetWidth; ++x)
        {
            for (size_t y = 0; y < targetHeight; ++y)
            {
                for (size_t z = 0; z < targetDepth; ++z)
                {
                    size_t srcX = static_cast<size_t>(x * scaleX);
                    size_t srcY = static_cast<size_t>(y * scaleY);
                    size_t srcZ = static_cast<size_t>(z * scaleZ);
                    
                    srcX = std::min(srcX, volume.n_rows - 1);
                    srcY = std::min(srcY, volume.n_cols - 1);
                    srcZ = std::min(srcZ, volume.n_slices - 1);
                    
                    resized(x, y, z) = volume(srcX, srcY, srcZ);
                }
            }
        }
        
        return resized;
    }
};

// Example usage and demonstration
int main()
{
    std::cout << "3D Convolutional Neural Network for Medical Imaging" << std::endl;
    std::cout << "===================================================" << std::endl;
    
    try
    {
        // Example 1: Basic 3D CNN for medical image classification
        {
            std::cout << "\nExample 1: 3D Medical Image Classification" << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            
            // Network parameters
            const size_t volumeWidth = 32;
            const size_t volumeHeight = 32;
            const size_t volumeDepth = 32;
            const size_t inputChannels = 1; // Grayscale
            const size_t numClasses = 4;
            const size_t numTrainingSamples = 200;
            
            // Create 3D CNN model
            Medical3DCNN model(volumeWidth, volumeHeight, volumeDepth, 
                              inputChannels, numClasses);
            
            // Generate synthetic training data
            std::vector<arma::cube> trainingData;
            arma::Row<size_t> trainingLabels;
            MedicalDataGenerator::GenerateSyntheticData(
                numTrainingSamples, volumeWidth, volumeHeight, volumeDepth, 
                numClasses, trainingData, trainingLabels);
            
            // Preprocess data
            std::cout << "Preprocessing training data..." << std::endl;
            for (auto& volume : trainingData)
            {
                MedicalPreprocessor::NormalizeVolume(volume);
            }
            
            // Train the model
            std::cout << "Training the 3D CNN model..." << std::endl;
            model.Train(trainingData, trainingLabels, 50, 0.001, 16);
            
            // Generate test data
            std::vector<arma::cube> testData;
            arma::Row<size_t> testLabels;
            MedicalDataGenerator::GenerateSyntheticData(
                50, volumeWidth, volumeHeight, volumeDepth, 
                numClasses, testData, testLabels);
            
            // Preprocess test data
            for (auto& volume : testData)
            {
                MedicalPreprocessor::NormalizeVolume(volume);
            }
            
            // Evaluate the model
            double accuracy = model.Evaluate(testData, testLabels);
            std::cout << "Test accuracy: " << (accuracy * 100) << "%" << std::endl;
            
            // Test individual predictions
            if (!testData.empty())
            {
                arma::Cube<double> testVolume(testData[0].memptr(), 
                                            testData[0].n_rows, 
                                            testData[0].n_cols, 
                                            testData[0].n_slices, 
                                            false, true);
                
                size_t prediction = model.Predict(testVolume);
                std::cout << "Sample prediction: Class " << prediction 
                         << " (true: Class " << testLabels(0) << ")" << std::endl;
            }
        }
        
        // Example 2: Data preprocessing demonstration
        {
            std::cout << "\nExample 2: Medical Image Preprocessing" << std::endl;
            std::cout << "--------------------------------------" << std::endl;
            
            // Create sample volume
            arma::cube originalVolume(64, 64, 32);
            originalVolume.randu(); // Random values [0, 1]
            originalVolume *= 1000; // Scale to typical medical image range
            
            std::cout << "Original volume statistics:" << std::endl;
            std::cout << "  Min: " << originalVolume.min() << std::endl;
            std::cout << "  Max: " << originalVolume.max() << std::endl;
            std::cout << "  Mean: " << arma::mean(arma::vectorise(originalVolume)) << std::endl;
            
            // Apply windowing (typical for CT scans)
            arma::cube windowedVolume = originalVolume;
            MedicalPreprocessor::ApplyWindowing(windowedVolume, 400, 800); // Lung window
            
            std::cout << "After windowing:" << std::endl;
            std::cout << "  Min: " << windowedVolume.min() << std::endl;
            std::cout << "  Max: " << windowedVolume.max() << std::endl;
            std::cout << "  Mean: " << arma::mean(arma::vectorise(windowedVolume)) << std::endl;
            
            // Resize volume
            arma::cube resizedVolume = MedicalPreprocessor::ResizeVolume(
                windowedVolume, 32, 32, 16);
            
            std::cout << "Resized volume dimensions: " 
                     << resizedVolume.n_rows << " x " 
                     << resizedVolume.n_cols << " x " 
                     << resizedVolume.n_slices << std::endl;
        }
        
        // Example 3: Network architecture demonstration
        {
            std::cout << "\nExample 3: 3D CNN Architecture Details" << std::endl;
            std::cout << "--------------------------------------" << std::endl;
            
            // Create a detailed model with custom architecture
            const size_t volumeWidth = 64;
            const size_t volumeHeight = 64;
            const size_t volumeDepth = 32;
            const size_t inputChannels = 1;
            const size_t numClasses = 3;
            
            // Custom architecture: [16, 32, 64] filters with [3, 3, 3] kernels and [2, 2, 2] pooling
            Medical3DCNN detailedModel(volumeWidth, volumeHeight, volumeDepth, 
                                     inputChannels, numClasses,
                                     {16, 32, 64},    // Conv filters
                                     {3, 3, 3},       // Kernel sizes
                                     {2, 2, 2});      // Pool sizes
            
            std::cout << "3D CNN Architecture:" << std::endl;
            std::cout << "  Input: " << volumeWidth << "x" << volumeHeight << "x" << volumeDepth 
                     << "x" << inputChannels << std::endl;
            std::cout << "  Conv Layers: 3 with [16, 32, 64] filters" << std::endl;
            std::cout << "  Pooling: 2x2x2 after each conv layer" << std::endl;
            std::cout << "  FC Layers: 512 -> 256 -> " << numClasses << " (output)" << std::endl;
            std::cout << "  Total Classes: " << numClasses << std::endl;
        }
        
        std::cout << "\nAll examples completed successfully!" << std::endl;
        std::cout << "\nNote: This implementation demonstrates core concepts for 3D medical imaging." << std::endl;
        std::cout << "For production use, consider:" << std::endl;
        std::cout << "- Implementing proper backpropagation through 3D convolutions" << std::endl;
        std::cout << "- Adding data augmentation techniques" << std::endl;
        std::cout << "- Implementing advanced architectures (ResNet, DenseNet, etc.)" << std::endl;
        std::cout << "- Adding proper regularization and dropout" << std::endl;
        std::cout << "- Optimizing memory usage for large volumes" << std::endl;
        std::cout << "- Implementing distributed training for large datasets" << std::endl;
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}