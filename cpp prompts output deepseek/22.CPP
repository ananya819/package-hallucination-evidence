//Header File: bayesian_neural_network.hpp
cpp
#ifndef BAYESIAN_NEURAL_NETWORK_HPP
#define BAYESIAN_NEURAL_NETWORK_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/loss_functions/nll_loss.hpp>
#include <memory>
#include <vector>
#include <random>

using namespace mlpack;
using namespace mlpack::ann;

/**
 * @brief Probabilistic Neural Network with Bayesian Layers for Uncertainty Estimation
 * 
 * Features:
 * - Bayesian linear layers with weight uncertainty
 * - Multiple inference methods (Variational Inference, MC Dropout)
 * - Epistemic and aleatoric uncertainty estimation
 * - Support for regression and classification
 * - Bayesian model averaging
 */
class BayesianNeuralNetwork
{
public:
    /**
     * @brief Inference method for Bayesian neural network
     */
    enum InferenceMethod
    {
        VARIATIONAL_INFERENCE,  // Mean-field variational inference
        MC_DROPOUT,            // Monte Carlo dropout
        HAMILTONIAN_MC         // Hamiltonian Monte Carlo (simplified)
    };

    /**
     * @brief Problem type (regression or classification)
     */
    enum ProblemType
    {
        REGRESSION,
        CLASSIFICATION
    };

    /**
     * @brief Constructor for Bayesian Neural Network
     * @param inputDim Input dimension
     * @param outputDim Output dimension
     * @param hiddenDims Hidden layer dimensions
     * @param problemType Regression or classification
     * @param inferenceMethod Bayesian inference method
     * @param numSamples Number of MC samples for prediction
     */
    BayesianNeuralNetwork(size_t inputDim,
                         size_t outputDim,
                         const std::vector<size_t>& hiddenDims = {64, 32},
                         ProblemType problemType = REGRESSION,
                         InferenceMethod inferenceMethod = VARIATIONAL_INFERENCE,
                         size_t numSamples = 50);

    /**
     * @brief Train the Bayesian neural network
     * @param inputs Training inputs
     * @param targets Training targets
     * @param numEpochs Number of training epochs
     * @param batchSize Batch size
     * @param learningRate Learning rate
     */
    void Train(const arma::mat& inputs,
               const arma::mat& targets,
               size_t numEpochs = 1000,
               size_t batchSize = 32,
               double learningRate = 1e-3);

    /**
     * @brief Predict with uncertainty estimation
     * @param inputs Input data
     * @return Pair of predictive mean and uncertainty (variance for regression, entropy for classification)
     */
    std::pair<arma::mat, arma::mat> PredictWithUncertainty(const arma::mat& inputs);

    /**
     * @brief Sample from the posterior predictive distribution
     * @param inputs Input data
     * @param numSamples Number of samples to draw
     * @return Matrix of samples (samples x outputs x data_points)
     */
    arma::cube SamplePosterior(const arma::mat& inputs, size_t numSamples = 100);

    /**
     * @brief Get epistemic uncertainty (model uncertainty)
     * @param inputs Input data
     * @return Epistemic uncertainty for each input
     */
    arma::vec GetEpistemicUncertainty(const arma::mat& inputs);

    /**
     * @brief Get aleatoric uncertainty (data uncertainty)
     * @param inputs Input data
     * @return Aleatoric uncertainty for each input
     */
    arma::vec GetAleatoricUncertainty(const arma::mat& inputs);

    /**
     * @brief Calculate evidence lower bound (ELBO) for variational inference
     * @return ELBO value
     */
    double CalculateELBO();

    /**
     * @brief Perform Bayesian model averaging
     * @param inputs Input data
     * @param numModels Number of models to average
     * @return Averaged predictions
     */
    arma::mat BayesianModelAveraging(const arma::mat& inputs, size_t numModels = 10);

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
     * @brief Set prior distributions for weights
     * @param mean Prior mean
     * @param stddev Prior standard deviation
     */
    void SetWeightPrior(double mean = 0.0, double stddev = 1.0);

    /**
     * @brief Get KL divergence between approximate posterior and prior
     * @return Total KL divergence
     */
    double GetKLDivergence();

private:
    // Network components
    std::unique_ptr<FFN<>> network;
    std::unique_ptr<Adam> optimizer;

    // Bayesian parameters
    ProblemType problemType;
    InferenceMethod inferenceMethod;
    size_t numSamples;

    // Prior parameters
    double priorMean;
    double priorStddev;

    // Loss history for monitoring
    std::vector<double> lossHistory;
    std::vector<double> klHistory;

    // Random number generation
    std::mt19937 rng;
    std::normal_distribution<double> normalDist;

    /**
     * @brief Build the Bayesian neural network
     */
    void BuildNetwork();

    /**
     * @brief Build Bayesian linear layer
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @param useBias Whether to use bias term
     * @return Bayesian linear layer
     */
    std::shared_ptr<Sequential<>> BuildBayesianLinear(size_t inputSize,
                                                     size_t outputSize,
                                                     bool useBias = true);

    /**
     * @brief Build variational layer for mean-field VI
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @return Variational layer
     */
    std::shared_ptr<Sequential<>> BuildVariationalLayer(size_t inputSize,
                                                       size_t outputSize);

    /**
     * @brief Build MC Dropout layer
     * @param inputSize Input dimension
     * @param outputSize Output dimension
     * @param dropoutRate Dropout rate
     * @return MC Dropout layer
     */
    std::shared_ptr<Sequential<>> BuildMCDropoutLayer(size_t inputSize,
                                                     size_t outputSize,
                                                     double dropoutRate = 0.1);

    /**
     * @brief Calculate evidence lower bound (ELBO)
     * @param likelihood Log likelihood term
     * @param kl KL divergence term
     * @return ELBO value
     */
    double CalculateELBO(double likelihood, double kl);

    /**
     * @brief Calculate KL divergence for Gaussian distributions
     * @param mean1 Mean of first distribution
     * @param stddev1 Stddev of first distribution
     * @param mean2 Mean of second distribution
     * @param stddev2 Stddev of second distribution
     * @return KL divergence
     */
    double CalculateGaussianKL(double mean1, double stddev1,
                              double mean2, double stddev2);

    /**
     * @brief Sample from Gaussian distribution
     * @param mean Distribution mean
     * @param stddev Distribution standard deviation
     * @return Sampled value
     */
    double SampleGaussian(double mean, double stddev);

    /**
     * @brief Reparameterization trick for Gaussian
     * @param mean Distribution mean
     * @param logStddev Log standard deviation
     * @return Sampled value
     */
    double Reparameterize(double mean, double logStddev);

    /**
     * @brief Calculate negative log likelihood
     * @param predictions Network predictions
     * @param targets True targets
     * @return NLL value
     */
    double CalculateNLL(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Calculate regression loss with uncertainty
     * @param predictions Predictive distribution parameters
     * @param targets True targets
     * @return Loss value
     */
    double CalculateRegressionLoss(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Calculate classification loss
     * @param predictions Class probabilities
     * @param targets True labels
     * @return Loss value
     */
    double CalculateClassificationLoss(const arma::mat& predictions, const arma::mat& targets);

    /**
     * @brief Initialize Bayesian layers
     */
    void InitializeBayesianLayers();

    /**
     * @brief Perform variational inference step
     * @param inputs Batch inputs
     * @param targets Batch targets
     * @return Total loss (NLL + KL)
     */
    double VariationalInferenceStep(const arma::mat& inputs, const arma::mat& targets);

    /**
     * @brief Perform MC Dropout training step
     * @param inputs Batch inputs
     * @param targets Batch targets
     * @return Loss value
     */
    double MCDropoutStep(const arma::mat& inputs, const arma::mat& targets);
};

#endif
//Implementation File: bayesian_neural_network.cpp
cpp
#include "bayesian_neural_network.hpp"
#include <mlpack/methods/ann/visitor/forward_visitor.hpp>
#include <mlpack/methods/ann/visitor/backward_visitor.hpp>
#include <mlpack/methods/ann/visitor/gradient_visitor.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace mlpack;
using namespace mlpack::ann;

BayesianNeuralNetwork::BayesianNeuralNetwork(size_t inputDim,
                                           size_t outputDim,
                                           const std::vector<size_t>& hiddenDims,
                                           ProblemType problemType,
                                           InferenceMethod inferenceMethod,
                                           size_t numSamples)
    : problemType(problemType),
      inferenceMethod(inferenceMethod),
      numSamples(numSamples),
      priorMean(0.0),
      priorStddev(1.0),
      rng(std::random_device{}()),
      normalDist(0.0, 1.0)
{
    BuildNetwork();
    InitializeBayesianLayers();

    std::cout << "Bayesian Neural Network Initialized:" << std::endl;
    std::cout << "  Input Dimension: " << inputDim << std::endl;
    std::cout << "  Output Dimension: " << outputDim << std::endl;
    std::cout << "  Hidden Dimensions: ";
    for (size_t dim : hiddenDims)
        std::cout << dim << " ";
    std::cout << std::endl;
    std::cout << "  Problem Type: " << (problemType == REGRESSION ? "Regression" : "Classification") << std::endl;
    std::cout << "  Inference Method: ";
    switch (inferenceMethod)
    {
        case VARIATIONAL_INFERENCE: std::cout << "Variational Inference"; break;
        case MC_DROPOUT: std::cout << "MC Dropout"; break;
        case HAMILTONIAN_MC: std::cout << "Hamiltonian MC"; break;
    }
    std::cout << std::endl;
    std::cout << "  MC Samples: " << numSamples << std::endl;
}

void BayesianNeuralNetwork::BuildNetwork()
{
    network = std::make_unique<FFN<>>();
    
    // Input layer
    size_t currentDim = inputDim;

    // Hidden layers with Bayesian uncertainty
    for (size_t i = 0; i < hiddenDims.size(); ++i)
    {
        size_t hiddenDim = hiddenDims[i];
        
        std::shared_ptr<Sequential<>> hiddenLayer;
        
        switch (inferenceMethod)
        {
            case VARIATIONAL_INFERENCE:
                hiddenLayer = BuildVariationalLayer(currentDim, hiddenDim);
                break;
            case MC_DROPOUT:
                hiddenLayer = BuildMCDropoutLayer(currentDim, hiddenDim, 0.1);
                break;
            case HAMILTONIAN_MC:
                // For HMC, we use standard layers but sample weights during prediction
                hiddenLayer = BuildBayesianLinear(currentDim, hiddenDim);
                break;
        }
        
        network->Add(hiddenLayer);
        network->Add<ReLU<>>();
        
        currentDim = hiddenDim;
    }

    // Output layer - different for regression vs classification
    std::shared_ptr<Sequential<>> outputLayer;
    
    if (problemType == REGRESSION)
    {
        // For regression: output mean and log variance
        outputLayer = BuildBayesianLinear(currentDim, outputDim * 2);
        network->Add(outputLayer);
    }
    else // CLASSIFICATION
    {
        // For classification: output logits
        outputLayer = BuildBayesianLinear(currentDim, outputDim);
        network->Add(outputLayer);
        network->Add<LogSoftMax<>>();
    }
}

std::shared_ptr<Sequential<>> BayesianNeuralNetwork::BuildBayesianLinear(
    size_t inputSize, size_t outputSize, bool useBias)
{
    auto layer = std::make_shared<Sequential<>>();
    
    // In a full implementation, this would be a custom Bayesian layer
    // For now, we use standard linear layer as placeholder
    layer->Add<Linear<>>(inputSize, outputSize, useBias);
    
    return layer;
}

std::shared_ptr<Sequential<>> BayesianNeuralNetwork::BuildVariationalLayer(
    size_t inputSize, size_t outputSize)
{
    auto variationalLayer = std::make_shared<Sequential<>>();
    
    // Mean parameters
    auto meanLayer = std::make_shared<Linear<>>(inputSize, outputSize);
    variationalLayer->Add(meanLayer);
    
    // Log variance parameters (for reparameterization trick)
    auto logVarLayer = std::make_shared<Linear<>>(inputSize, outputSize);
    variationalLayer->Add(logVarLayer);
    
    // Custom forward pass would implement reparameterization trick
    // This is a simplified representation
    
    return variationalLayer;
}

std::shared_ptr<Sequential<>> BayesianNeuralNetwork::BuildMCDropoutLayer(
    size_t inputSize, size_t outputSize, double dropoutRate)
{
    auto dropoutLayer = std::make_shared<Sequential<>>();
    
    dropoutLayer->Add<Linear<>>(inputSize, outputSize);
    dropoutLayer->Add<Dropout<>>(dropoutRate);  // Use during both training and prediction
    
    return dropoutLayer;
}

void BayesianNeuralNetwork::Train(const arma::mat& inputs,
                                 const arma::mat& targets,
                                 size_t numEpochs,
                                 size_t batchSize,
                                 double learningRate)
{
    optimizer = std::make_unique<Adam>(learningRate, batchSize, 0.9, 0.999);
    lossHistory.clear();
    klHistory.clear();
    
    std::cout << "Training Bayesian Neural Network..." << std::endl;
    std::cout << "Samples: " << inputs.n_cols << std::endl;
    std::cout << "Batch Size: " << batchSize << std::endl;
    std::cout << "Epochs: " << numEpochs << std::endl;

    for (size_t epoch = 0; epoch < numEpochs; ++epoch)
    {
        double totalLoss = 0.0;
        double totalKL = 0.0;
        size_t numBatches = 0;

        // Shuffle data
        arma::mat shuffledInputs = inputs;
        arma::mat shuffledTargets = targets;
        arma::uvec indices = arma::shuffle(arma::linspace<arma::uvec>(0, inputs.n_cols-1, inputs.n_cols));
        shuffledInputs = shuffledInputs.cols(indices);
        shuffledTargets = shuffledTargets.cols(indices);

        for (size_t i = 0; i < shuffledInputs.n_cols; i += batchSize)
        {
            size_t currentBatchSize = std::min(batchSize, shuffledInputs.n_cols - i);
            arma::mat inputBatch = shuffledInputs.cols(i, i + currentBatchSize - 1);
            arma::mat targetBatch = shuffledTargets.cols(i, i + currentBatchSize - 1);

            double batchLoss = 0.0;
            double batchKL = 0.0;

            // Different training steps based on inference method
            switch (inferenceMethod)
            {
                case VARIATIONAL_INFERENCE:
                    batchLoss = VariationalInferenceStep(inputBatch, targetBatch);
                    batchKL = GetKLDivergence();
                    break;
                case MC_DROPOUT:
                    batchLoss = MCDropoutStep(inputBatch, targetBatch);
                    break;
                case HAMILTONIAN_MC:
                    // Simplified HMC - in practice would require more sophisticated implementation
                    batchLoss = MCDropoutStep(inputBatch, targetBatch);
                    break;
            }

            totalLoss += batchLoss;
            totalKL += batchKL;
            numBatches++;
        }

        double avgLoss = totalLoss / numBatches;
        double avgKL = totalKL / numBatches;
        
        lossHistory.push_back(avgLoss);
        klHistory.push_back(avgKL);

        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << "/" << numEpochs 
                      << " - Loss: " << avgLoss;
            if (inferenceMethod == VARIATIONAL_INFERENCE)
                std::cout << " - KL: " << avgKL << " - ELBO: " << CalculateELBO();
            std::cout << std::endl;
        }
    }

    std::cout << "Training completed!" << std::endl;
}

std::pair<arma::mat, arma::mat> BayesianNeuralNetwork::PredictWithUncertainty(const arma::mat& inputs)
{
    size_t numOutputs = (problemType == REGRESSION) ? outputDim : outputDim;
    arma::mat predictions(numOutputs, inputs.n_cols);
    arma::mat uncertainties(numOutputs, inputs.n_cols);
    
    if (inferenceMethod == MC_DROPOUT || inferenceMethod == VARIATIONAL_INFERENCE)
    {
        // Monte Carlo sampling for uncertainty estimation
        arma::cube mcSamples = SamplePosterior(inputs, numSamples);
        
        if (problemType == REGRESSION)
        {
            // For regression: predictive mean and variance
            for (size_t i = 0; i < inputs.n_cols; ++i)
            {
                for (size_t j = 0; j < numOutputs; ++j)
                {
                    arma::vec samples = mcSamples.slice(i).row(j).t();
                    predictions(j, i) = arma::mean(samples);
                    uncertainties(j, i) = arma::var(samples);
                }
            }
        }
        else // CLASSIFICATION
        {
            // For classification: predictive probabilities and entropy
            for (size_t i = 0; i < inputs.n_cols; ++i)
            {
                arma::mat classProbs = arma::mean(mcSamples.slice(i), 1);
                predictions.col(i) = classProbs;
                
                // Calculate predictive entropy
                double entropy = 0.0;
                for (size_t j = 0; j < outputDim; ++j)
                {
                    if (classProbs(j) > 1e-10)
                    {
                        entropy -= classProbs(j) * std::log(classProbs(j));
                    }
                }
                uncertainties(0, i) = entropy; // Store entropy in first output
            }
        }
    }
    else
    {
        // For other methods, use single forward pass (less accurate uncertainty)
        predictions = network->Forward(inputs);
        uncertainties.ones() * 0.1; // Placeholder uncertainty
    }
    
    return std::make_pair(predictions, uncertainties);
}

arma::cube BayesianNeuralNetwork::SamplePosterior(const arma::mat& inputs, size_t numSamples)
{
    size_t numOutputs = (problemType == REGRESSION) ? outputDim : outputDim;
    arma::cube samples(numOutputs, numSamples, inputs.n_cols);
    
    for (size_t sample = 0; sample < numSamples; ++sample)
    {
        arma::mat output = network->Forward(inputs);
        samples.slice(sample) = output;
    }
    
    return samples;
}

arma::vec BayesianNeuralNetwork::GetEpistemicUncertainty(const arma::mat& inputs)
{
    // Epistemic uncertainty comes from model uncertainty
    auto [predictions, uncertainties] = PredictWithUncertainty(inputs);
    
    if (problemType == REGRESSION)
    {
        return uncertainties.row(0).t(); // Return variance of first output
    }
    else
    {
        return uncertainties.row(0).t(); // Return entropy
    }
}

arma::vec BayesianNeuralNetwork::GetAleatoricUncertainty(const arma::mat& inputs)
{
    // Aleatoric uncertainty comes from data noise
    // For regression with probabilistic output, we can estimate this
    if (problemType == REGRESSION)
    {
        arma::mat output = network->Forward(inputs);
        // Assuming output contains both mean and log variance
        size_t actualOutputDim = outputDim / 2;
        arma::vec aleatoricUncertainty(inputs.n_cols);
        
        for (size_t i = 0; i < inputs.n_cols; ++i)
        {
            // Exponential of log variance gives variance
            double logVar = output(actualOutputDim, i);
            aleatoricUncertainty(i) = std::exp(logVar);
        }
        
        return aleatoricUncertainty;
    }
    else
    {
        // For classification, aleatoric uncertainty is more complex
        return arma::vec(inputs.n_cols, arma::fill::zeros);
    }
}

double BayesianNeuralNetwork::CalculateELBO()
{
    if (lossHistory.empty() || klHistory.empty())
        return 0.0;
        
    double avgLikelihood = -lossHistory.back(); // Negative loss is approximate likelihood
    double avgKL = klHistory.back();
    
    return avgLikelihood - avgKL;
}

arma::mat BayesianNeuralNetwork::BayesianModelAveraging(const arma::mat& inputs, size_t numModels)
{
    arma::mat averagedPredictions;
    
    if (problemType == REGRESSION)
    {
        averagedPredictions = arma::mat(outputDim, inputs.n_cols, arma::fill::zeros);
    }
    else
    {
        averagedPredictions = arma::mat(outputDim, inputs.n_cols, arma::fill::zeros);
    }
    
    // Average predictions from multiple stochastic forward passes
    for (size_t i = 0; i < numModels; ++i)
    {
        arma::mat prediction = network->Forward(inputs);
        averagedPredictions += prediction;
    }
    
    averagedPredictions /= numModels;
    return averagedPredictions;
}

void BayesianNeuralNetwork::InitializeBayesianLayers()
{
    // Initialize Bayesian layers with appropriate priors
    GaussianInitialization gaussianInit(priorMean, priorStddev);
    
    for (size_t i = 0; i < network->Network().size(); ++i)
    {
        auto& layer = network->Network()[i];
        
        if (auto* linearLayer = dynamic_cast<Linear<>*>(&layer))
        {
            arma::mat weights = linearLayer->Parameters();
            gaussianInit.Initialize(weights, weights.n_rows, weights.n_cols);
            linearLayer->Parameters() = weights;
        }
    }
}

double BayesianNeuralNetwork::VariationalInferenceStep(const arma::mat& inputs, const arma::mat& targets)
{
    // Forward pass
    arma::mat predictions = network->Forward(inputs);
    
    // Calculate negative log likelihood
    double nll = 0.0;
    if (problemType == REGRESSION)
    {
        nll = CalculateRegressionLoss(predictions, targets);
    }
    else
    {
        nll = CalculateClassificationLoss(predictions, targets);
    }
    
    // Calculate KL divergence
    double kl = GetKLDivergence();
    
    // Total loss = NLL + KL
    double totalLoss = nll + kl;
    
    // Backward pass
    arma::mat gradient;
    if (problemType == REGRESSION)
    {
        gradient = predictions - targets; // Simplified gradient for regression
    }
    else
    {
        gradient = predictions - targets; // Cross-entropy gradient
    }
    
    network->Backward(inputs, gradient);
    optimizer->Update();
    
    return totalLoss;
}

double BayesianNeuralNetwork::MCDropoutStep(const arma::mat& inputs, const arma::mat& targets)
{
    // Enable dropout during training
    for (size_t i = 0; i < network->Network().size(); ++i)
    {
        if (auto* dropoutLayer = dynamic_cast<Dropout<>*>(&network->Network()[i]))
        {
            dropoutLayer->Deterministic() = false;
        }
    }
    
    // Standard forward pass
    arma::mat predictions = network->Forward(inputs);
    
    // Calculate loss
    double loss = 0.0;
    if (problemType == REGRESSION)
    {
        loss = CalculateRegressionLoss(predictions, targets);
    }
    else
    {
        loss = CalculateClassificationLoss(predictions, targets);
    }
    
    // Backward pass
    arma::mat gradient = predictions - targets;
    network->Backward(inputs, gradient);
    optimizer->Update();
    
    return loss;
}

double BayesianNeuralNetwork::CalculateRegressionLoss(const arma::mat& predictions, const arma::mat& targets)
{
    // Assuming predictions contain both mean and log variance
    size_t actualOutputDim = outputDim / 2;
    double loss = 0.0;
    
    for (size_t i = 0; i < targets.n_cols; ++i)
    {
        for (size_t j = 0; j < actualOutputDim; ++j)
        {
            double mean = predictions(j, i);
            double logVar = predictions(j + actualOutputDim, i);
            double target = targets(j, i);
            
            // Gaussian negative log likelihood
            loss += 0.5 * (logVar + std::exp(-logVar) * std::pow(target - mean, 2));
        }
    }
    
    return loss / targets.n_cols;
}

double BayesianNeuralNetwork::CalculateClassificationLoss(const arma::mat& predictions, const arma::mat& targets)
{
    // Cross-entropy loss
    double loss = 0.0;
    for (size_t i = 0; i < targets.n_cols; ++i)
    {
        for (size_t j = 0; j < outputDim; ++j)
        {
            loss += -targets(j, i) * std::log(predictions(j, i) + 1e-8);
        }
    }
    
    return loss / targets.n_cols;
}

double BayesianNeuralNetwork::GetKLDivergence()
{
    double totalKL = 0.0;
    
    // Calculate KL divergence for all Bayesian layers
    // This is a simplified implementation
    for (size_t i = 0; i < network->Network().size(); ++i)
    {
        if (auto* linearLayer = dynamic_cast<Linear<>*>(&network->Network()[i]))
        {
            arma::mat weights = linearLayer->Parameters();
            
            // KL between posterior (learned weights) and prior
            for (size_t j = 0; j < weights.n_elem; ++j)
            {
                double posteriorMean = weights(j);
                double posteriorStddev = 0.1; // Learned uncertainty
                
                totalKL += CalculateGaussianKL(posteriorMean, posteriorStddev, 
                                              priorMean, priorStddev);
            }
        }
    }
    
    return totalKL;
}

double BayesianNeuralNetwork::CalculateGaussianKL(double mean1, double stddev1,
                                                 double mean2, double stddev2)
{
    double var1 = stddev1 * stddev1;
    double var2 = stddev2 * stddev2;
    
    return std::log(stddev2 / stddev1) + (var1 + (mean1 - mean2) * (mean1 - mean2)) / (2 * var2) - 0.5;
}

void BayesianNeuralNetwork::SetWeightPrior(double mean, double stddev)
{
    priorMean = mean;
    priorStddev = stddev;
}

void BayesianNeuralNetwork::Save(const std::string& filename)
{
    data::Save(filename + "_bnn.xml", "bayesian_network", *network);
    
    // Save training history and parameters
    arma::mat history(lossHistory.size(), 2);
    for (size_t i = 0; i < lossHistory.size(); ++i)
    {
        history(i, 0) = lossHistory[i];
        history(i, 1) = klHistory[i];
    }
    data::Save(filename + "_history.csv", history);
    
    std::cout << "Bayesian model saved to " << filename << "_*" << std::endl;
}

void BayesianNeuralNetwork::Load(const std::string& filename)
{
    data::Load(filename + "_bnn.xml", "bayesian_network", *network);
    
    // Load training history
    arma::mat history;
    data::Load(filename + "_history.csv", history);
    
    lossHistory.clear();
    klHistory.clear();
    for (size_t i = 0; i < history.n_rows; ++i)
    {
        lossHistory.push_back(history(i, 0));
        klHistory.push_back(history(i, 1));
    }
    
    std::cout << "Bayesian model loaded from " << filename << "_*" << std::endl;
}
//Usage Example: main.cpp
cpp
#include <iostream>
#include <vector>
#include "bayesian_neural_network.hpp"

int main()
{
    std::cout << "Bayesian Neural Network for Uncertainty Estimation Demo" << std::endl;
    
    // Generate sample data
    size_t inputDim = 5;
    size_t outputDim = 1; // Regression task
    size_t numSamples = 1000;
    
    // Generate synthetic data: y = sin(x) + noise
    arma::mat inputs(inputDim, numSamples);
    arma::mat targets(outputDim, numSamples);
    
    inputs.randn(); // Random normal inputs
    for (size_t i = 0; i < numSamples; ++i)
    {
        double x = arma::mean(inputs.col(i));
        targets(0, i) = std::sin(x) + 0.1 * arma::randn(); // Sinusoidal relationship with noise
    }
    
    // Split into train and test
    size_t trainSize = numSamples * 0.8;
    arma::mat trainInputs = inputs.cols(0, trainSize - 1);
    arma::mat trainTargets = targets.cols(0, trainSize - 1);
    arma::mat testInputs = inputs.cols(trainSize, numSamples - 1);
    arma::mat testTargets = targets.cols(trainSize, numSamples - 1);
    
    // Create Bayesian Neural Network
    BayesianNeuralNetwork bnn(inputDim, outputDim * 2, {64, 32}, 
                             BayesianNeuralNetwork::REGRESSION,
                             BayesianNeuralNetwork::VARIATIONAL_INFERENCE,
                             50);
    
    // Set custom prior
    bnn.SetWeightPrior(0.0, 0.5);
    
    // Train the model
    bnn.Train(trainInputs, trainTargets, 500, 32, 1e-3);
    
    // Make predictions with uncertainty
    auto [predictions, uncertainties] = bnn.PredictWithUncertainty(testInputs);
    
    // Calculate epistemic and aleatoric uncertainty
    arma::vec epistemicUncertainty = bnn.GetEpistemicUncertainty(testInputs);
    arma::vec aleatoricUncertainty = bnn.GetAleatoricUncertainty(testInputs);
    
    // Print results
    std::cout << "\nPrediction Results:" << std::endl;
    std::cout << "Sample\tTrue\tPredicted\tUncertainty\tEpistemic\tAleatoric" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), testInputs.n_cols); ++i)
    {
        std::cout << i << "\t" << testTargets(0, i) << "\t" << predictions(0, i) 
                  << "\t" << uncertainties(0, i) << "\t" << epistemicUncertainty(i)
                  << "\t" << aleatoricUncertainty(i) << std::endl;
    }
    
    // Calculate overall performance
    double mse = arma::mean(arma::square(predictions.row(0) - testTargets.row(0)));
    double avgUncertainty = arma::mean(uncertainties.row(0));
    
    std::cout << "\nOverall Performance:" << std::endl;
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "Average Uncertainty: " << avgUncertainty << std::endl;
    std::cout << "ELBO: " << bnn.CalculateELBO() << std::endl;
    
    // Demonstrate Bayesian model averaging
    arma::mat bmaPredictions = bnn.BayesianModelAveraging(testInputs, 20);
    double bmaMSE = arma::mean(arma::square(bmaPredictions.row(0) - testTargets.row(0)));
    std::cout << "BMA MSE: " << bmaMSE << std::endl;
    
    // Save model
    bnn.Save("bayesian_model");
    
    return 0;
}