#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <deque>
#include <queue>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * Leaky Integrate-and-Fire (LIF) Neuron Layer
 */
template<typename MatType = arma::mat>
class LIFLayer : public Layer<MatType>
{
public:
    LIFLayer(const size_t size,
             const double tau = 20.0,        // Membrane time constant
             const double threshold = 1.0,   // Firing threshold
             const double resetVoltage = 0.0, // Reset voltage
             const double refractoryPeriod = 5.0, // Refractory period
             const double dt = 1.0) :        // Time step
        Layer<MatType>(),
        size(size),
        tau(tau),
        threshold(threshold),
        resetVoltage(resetVoltage),
        refractoryPeriod(refractoryPeriod),
        dt(dt)
    {
        // Initialize membrane potentials
        membraneVoltage = arma::zeros<MatType>(size, 1);
        refractoryTime = arma::zeros<MatType>(size, 1);
        
        // Initialize synaptic weights (for input connections)
        weights.set_size(size, size);
        RandomInitialization().Initialize(weights, size, size);
        weights *= 0.1; // Small initial weights
        
        // Initialize spike history
        spikeHistory.resize(size);
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input shape: (input_size, batch_size) or (size, batch_size) for recurrent
        const size_t batchSize = input.n_cols;
        output.set_size(size, batchSize);
        output.zeros();
        
        // Store current spikes for STDP
        previousSpikes = currentSpikes;
        currentSpikes.set_size(size, batchSize);
        currentSpikes.zeros();
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t i = 0; i < size; ++i)
            {
                // Skip if neuron is in refractory period
                if (refractoryTime(i, b) > 0)
                {
                    refractoryTime(i, b) -= dt;
                    membraneVoltage(i, b) = resetVoltage;
                    continue;
                }
                
                // Update membrane potential (leaky integration)
                // dV/dt = -V/tau + input
                double inputCurrent = input(i, b);
                if (input.n_rows == size) // Recurrent input
                {
                    inputCurrent += arma::dot(weights.row(i), previousOutput.col(b));
                }
                
                membraneVoltage(i, b) = membraneVoltage(i, b) * 
                                      std::exp(-dt / tau) + inputCurrent;
                
                // Check for spike
                if (membraneVoltage(i, b) >= threshold)
                {
                    // Generate spike
                    output(i, b) = 1.0;
                    currentSpikes(i, b) = 1.0;
                    
                    // Reset membrane potential
                    membraneVoltage(i, b) = resetVoltage;
                    
                    // Set refractory period
                    refractoryTime(i, b) = refractoryPeriod;
                    
                    // Record spike time for STDP
                    spikeHistory[i].push_back(0); // Current time step
                }
                else
                {
                    output(i, b) = 0.0;
                }
            }
        }
        
        previousOutput = output;
        
        // Update spike history (age spikes)
        UpdateSpikeHistory();
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // SNNs typically use surrogate gradients for backpropagation
        // Here we implement a simple straight-through estimator
        g = gy;
    }

    /**
     * Update spike history (increment spike ages)
     */
    void UpdateSpikeHistory()
    {
        for (size_t i = 0; i < size; ++i)
        {
            for (auto& spikeTime : spikeHistory[i])
            {
                spikeTime++;
            }
            
            // Remove old spikes (keep only recent history for STDP)
            while (!spikeHistory[i].empty() && spikeHistory[i].front() > maxSTDPWindow)
            {
                spikeHistory[i].pop_front();
            }
        }
    }

    /**
     * Get membrane potentials for monitoring
     */
    const MatType& GetMembraneVoltage() const { return membraneVoltage; }

    /**
     * Get spike trains
     */
    const MatType& GetSpikes() const { return currentSpikes; }

    /**
     * Get weights for STDP
     */
    MatType& GetWeights() { return weights; }

    /**
     * Get spike history for STDP
     */
    const std::vector<std::deque<int>>& GetSpikeHistory() const { return spikeHistory; }

    virtual LIFLayer* Clone() const
    {
        return new LIFLayer(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(size));
        ar(CEREAL_NVP(tau));
        ar(CEREAL_NVP(threshold));
        ar(CEREAL_NVP(resetVoltage));
        ar(CEREAL_NVP(refractoryPeriod));
        ar(CEREAL_NVP(dt));
        ar(CEREAL_NVP(weights));
        ar(CEREAL_NVP(membraneVoltage));
        ar(CEREAL_NVP(refractoryTime));
    }

private:
    size_t size;
    double tau;
    double threshold;
    double resetVoltage;
    double refractoryPeriod;
    double dt;
    const int maxSTDPWindow = 100; // Maximum time window for STDP
    
    MatType weights;
    MatType membraneVoltage;
    MatType refractoryTime;
    MatType previousOutput;
    MatType currentSpikes;
    MatType previousSpikes;
    
    std::vector<std::deque<int>> spikeHistory; // Spike times for each neuron
};

/**
 * Spike-Timing-Dependent Plasticity (STDP) Learning Rule
 */
class STDPLearning
{
public:
    STDPLearning(const double aPlus = 0.01,   // LTP rate
                 const double aMinus = 0.01,  // LTD rate
                 const double tauPlus = 20.0, // LTP time constant
                 const double tauMinus = 20.0, // LTD time constant
                 const double maxWeight = 2.0, // Maximum weight
                 const double minWeight = 0.0) : // Minimum weight
        aPlus(aPlus),
        aMinus(aMinus),
        tauPlus(tauPlus),
        tauMinus(tauMinus),
        maxWeight(maxWeight),
        minWeight(minWeight)
    {}

    /**
     * Apply STDP learning rule
     */
    template<typename MatType>
    void Learn(LIFLayer<MatType>& preLayer, LIFLayer<MatType>& postLayer)
    {
        MatType& weights = preLayer.GetWeights();
        const std::vector<std::deque<int>>& preSpikeHistory = preLayer.GetSpikeHistory();
        const std::vector<std::deque<int>>& postSpikeHistory = postLayer.GetSpikeHistory();
        
        const size_t preSize = preSpikeHistory.size();
        const size_t postSize = postSpikeHistory.size();
        
        // Apply STDP for each connection
        for (size_t i = 0; i < preSize; ++i)
        {
            for (size_t j = 0; j < postSize; ++j)
            {
                double weightChange = 0.0;
                
                // Post-synaptic spike after pre-synaptic spike (LTP)
                for (int preTime : preSpikeHistory[i])
                {
                    for (int postTime : postSpikeHistory[j])
                    {
                        int deltaT = postTime - preTime;
                        
                        if (deltaT > 0) // Post after pre (LTP)
                        {
                            weightChange += aPlus * std::exp(-deltaT / tauPlus);
                        }
                        else if (deltaT < 0) // Pre after post (LTD)
                        {
                            weightChange -= aMinus * std::exp(deltaT / tauMinus);
                        }
                    }
                }
                
                // Update weight with bounds
                weights(j, i) += weightChange;
                weights(j, i) = std::max(minWeight, std::min(maxWeight, weights(j, i)));
            }
        }
    }

    /**
     * Apply STDP with pre and post spike trains
     */
    void ApplySTDP(arma::mat& weights,
                   const arma::mat& preSpikes,
                   const arma::mat& postSpikes,
                   const double learningRate = 1.0)
    {
        const size_t preSize = preSpikes.n_rows;
        const size_t postSize = postSpikes.n_rows;
        const size_t batchSize = preSpikes.n_cols;
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t i = 0; i < preSize; ++i)
            {
                for (size_t j = 0; j < postSize; ++j)
                {
                    if (preSpikes(i, b) > 0.5 && postSpikes(j, b) > 0.5)
                    {
                        // Simple STDP: if both fire simultaneously, strengthen connection
                        weights(j, i) += aPlus * learningRate;
                    }
                    else if (preSpikes(i, b) > 0.5 && postSpikes(j, b) < 0.5)
                    {
                        // Pre fires but post doesn't - weaken connection
                        weights(j, i) -= aMinus * learningRate;
                    }
                    
                    // Apply weight bounds
                    weights(j, i) = std::max(minWeight, std::min(maxWeight, weights(j, i)));
                }
            }
        }
    }

private:
    double aPlus;
    double aMinus;
    double tauPlus;
    double tauMinus;
    double maxWeight;
    double minWeight;
};

/**
 * Spiking Neural Network with STDP Learning
 */
class SpikingNeuralNetwork
{
public:
    SpikingNeuralNetwork(const size_t inputSize,
                         const size_t hiddenSize,
                         const size_t outputSize,
                         const double simulationTime = 100.0,
                         const double dt = 1.0) :
        inputSize(inputSize),
        hiddenSize(hiddenSize),
        outputSize(outputSize),
        simulationTime(simulationTime),
        dt(dt)
    {
        InitializeNetwork();
        InitializeSTDP();
    }

    void InitializeNetwork()
    {
        // Input encoding layer (convert analog to spikes)
        inputLayer = std::make_unique<LIFLayer<>>(inputSize, 10.0, 0.5, 0.0, 2.0, dt);
        
        // Hidden layer
        hiddenLayer = std::make_unique<LIFLayer<>>(hiddenSize, 20.0, 1.0, 0.0, 5.0, dt);
        
        // Output layer
        outputLayer = std::make_unique<LIFLayer<>>(outputSize, 20.0, 1.0, 0.0, 5.0, dt);
        
        std::cout << "SNN Initialized:" << std::endl;
        std::cout << "  Input neurons: " << inputSize << std::endl;
        std::cout << "  Hidden neurons: " << hiddenSize << std::endl;
        std::cout << "  Output neurons: " << outputSize << std::endl;
    }

    void InitializeSTDP()
    {
        stdp = std::make_unique<STDPLearning>(0.01, 0.01, 20.0, 20.0, 2.0, 0.0);
    }

    /**
     * Convert analog input to spike trains using rate coding
     */
    mat EncodeInput(const mat& analogInput)
    {
        mat spikeInput(inputSize, analogInput.n_cols);
        
        for (size_t i = 0; i < analogInput.n_elem; ++i)
        {
            // Rate coding: higher input values = higher firing probability
            double firingProb = std::min(1.0, std::max(0.0, analogInput(i)));
            spikeInput(i) = (arma::randu() < firingProb) ? 1.0 : 0.0;
        }
        
        return spikeInput;
    }

    /**
     * Run SNN simulation for a given input
     */
    mat Forward(const mat& input)
    {
        const size_t batchSize = input.n_cols;
        const size_t timeSteps = static_cast<size_t>(simulationTime / dt);
        
        // Initialize output spike counts
        mat outputSpikeCounts(outputSize, batchSize, arma::fill::zeros);
        
        for (size_t t = 0; t < timeSteps; ++t)
        {
            // Encode input at current time step
            mat currentInput = EncodeInput(input);
            
            // Forward pass through layers
            mat inputSpikes, hiddenSpikes, outputSpikes;
            
            inputLayer->Forward(currentInput, inputSpikes);
            hiddenLayer->Forward(inputSpikes, hiddenSpikes);
            outputLayer->Forward(hiddenSpikes, outputSpikes);
            
            // Accumulate output spikes
            outputSpikeCounts += outputSpikes;
            
            // Apply STDP learning (every few time steps)
            if (t % 10 == 0)
            {
                stdp->Learn(*inputLayer, *hiddenLayer);
                stdp->Learn(*hiddenLayer, *outputLayer);
            }
        }
        
        // Convert spike counts to firing rates
        mat firingRates = outputSpikeCounts / timeSteps;
        
        return firingRates;
    }

    /**
     * Train SNN on dataset
     */
    void Train(const mat& trainInputs, const mat& trainTargets,
               const size_t epochs = 100, const size_t batchSize = 32)
    {
        std::cout << "Training SNN with STDP..." << std::endl;
        std::cout << "Samples: " << trainInputs.n_cols << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            double totalError = 0.0;
            size_t numBatches = 0;
            
            for (size_t i = 0; i < trainInputs.n_cols; i += batchSize)
            {
                size_t currentBatchSize = std::min(batchSize, trainInputs.n_cols - i);
                
                mat batchInput = trainInputs.cols(i, i + currentBatchSize - 1);
                mat batchTarget = trainTargets.cols(i, i + currentBatchSize - 1);
                
                // Forward pass
                mat output = Forward(batchInput);
                
                // Calculate error (MSE between firing rates and targets)
                double error = arma::accu(arma::square(output - batchTarget)) / currentBatchSize;
                totalError += error;
                numBatches++;
                
                // STDP learning happens during forward pass
            }
            
            double avgError = totalError / numBatches;
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Avg Error: " << avgError << std::endl;
        }
    }

    /**
     * Get network state information
     */
    void PrintNetworkState()
    {
        std::cout << "\n=== SNN State ===" << std::endl;
        std::cout << "Input layer membrane potentials: " 
                  << arma::mean(arma::mean(inputLayer->GetMembraneVoltage())) << std::endl;
        std::cout << "Hidden layer membrane potentials: " 
                  << arma::mean(arma::mean(hiddenLayer->GetMembraneVoltage())) << std::endl;
        std::cout << "Output layer membrane potentials: " 
                  << arma::mean(arma::mean(outputLayer->GetMembraneVoltage())) << std::endl;
        
        std::cout << "Input->Hidden weights mean: " << arma::mean(arma::mean(hiddenLayer->GetWeights())) << std::endl;
        std::cout << "Hidden->Output weights mean: " << arma::mean(arma::mean(outputLayer->GetWeights())) << std::endl;
    }

    /**
     * Save network parameters
     */
    void SaveModel(const std::string& filename)
    {
        // Save weights and parameters
        data::Save(filename + "_input_weights.csv", inputLayer->GetWeights());
        data::Save(filename + "_hidden_weights.csv", hiddenLayer->GetWeights());
        data::Save(filename + "_output_weights.csv", outputLayer->GetWeights());
        std::cout << "SNN model saved to " << filename << std::endl;
    }

    /**
     * Load network parameters
     */
    void LoadModel(const std::string& filename)
    {
        // Load weights
        mat inputWeights, hiddenWeights, outputWeights;
        data::Load(filename + "_input_weights.csv", inputWeights);
        data::Load(filename + "_hidden_weights.csv", hiddenWeights);
        data::Load(filename + "_output_weights.csv", outputWeights);
        
        inputLayer->GetWeights() = inputWeights;
        hiddenLayer->GetWeights() = hiddenWeights;
        outputLayer->GetWeights() = outputWeights;
        
        std::cout << "SNN model loaded from " << filename << std::endl;
    }

private:
    size_t inputSize;
    size_t hiddenSize;
    size_t outputSize;
    double simulationTime;
    double dt;
    
    std::unique_ptr<LIFLayer<>> inputLayer;
    std::unique_ptr<LIFLayer<>> hiddenLayer;
    std::unique_ptr<LIFLayer<>> outputLayer;
    std::unique_ptr<STDPLearning> stdp;
};

/**
 * Example: MNIST digit classification with SNN
 */
class SNNMNISTClassifier
{
public:
    SNNMNISTClassifier(const size_t inputSize = 784,
                       const size_t hiddenSize = 100,
                       const size_t outputSize = 10) :
        snn(inputSize, hiddenSize, outputSize)
    {}

    void Train(const mat& trainData, const mat& trainLabels,
               const size_t epochs = 50)
    {
        std::cout << "Training SNN on MNIST data..." << std::endl;
        
        // Normalize input data to [0, 1] for rate coding
        mat normalizedData = trainData / 255.0;
        
        snn.Train(normalizedData, trainLabels, epochs, 32);
    }

    double Test(const mat& testData, const mat& testLabels)
    {
        size_t correct = 0;
        size_t total = testData.n_cols;
        
        mat normalizedData = testData / 255.0;
        
        for (size_t i = 0; i < total; ++i)
        {
            mat output = snn.Forward(normalizedData.col(i));
            
            uword predictedClass = output.index_max();
            uword trueClass = testLabels.col(i).index_max();
            
            if (predictedClass == trueClass)
                correct++;
        }
        
        double accuracy = (static_cast<double>(correct) / total) * 100.0;
        std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;
        return accuracy;
    }

private:
    SpikingNeuralNetwork snn;
};

/**
 * Example usage
 */
int main()
{
    // Create SNN for a simple classification task
    const size_t inputSize = 100;
    const size_t hiddenSize = 50;
    const size_t outputSize = 5;
    
    SpikingNeuralNetwork snn(inputSize, hiddenSize, outputSize);
    
    // Generate synthetic training data
    mat trainInputs = arma::randu<mat>(inputSize, 1000);
    mat trainTargets = arma::zeros<mat>(outputSize, 1000);
    
    // Create simple classification targets
    for (size_t i = 0; i < 1000; ++i)
    {
        size_t targetClass = i % outputSize;
        trainTargets(targetClass, i) = 1.0; // Target firing rate
    }
    
    // Train the SNN
    snn.Train(trainInputs, trainTargets, 10);
    
    // Print network state
    snn.PrintNetworkState();
    
    // Test the SNN
    mat testInput = arma::randu<mat>(inputSize, 1);
    mat output = snn.Forward(testInput);
    
    std::cout << "\nTest output firing rates: " << output.t() << std::endl;
    
    // Save the model
    snn.SaveModel("snn_model");
    
    return 0;
}