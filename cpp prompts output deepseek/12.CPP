#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/attention/attention.hpp>
#include <memory>
#include <deque>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * Multi-scale Hierarchical RNN Layer
 * Processes sequences at multiple temporal resolutions
 */
template<typename MatType = arma::mat>
class HierarchicalRNN : public Layer<MatType>
{
public:
    HierarchicalRNN(const size_t inputSize,
                    const size_t hiddenSize,
                    const size_t numLevels = 3,
                    const size_t rnnType = 0, // 0: LSTM, 1: GRU
                    const double dropoutRate = 0.2) :
        Layer<MatType>(),
        inputSize(inputSize),
        hiddenSize(hiddenSize),
        numLevels(numLevels),
        rnnType(rnnType),
        dropoutRate(dropoutRate)
    {
        InitializeHierarchy();
    }

    void InitializeHierarchy()
    {
        // Create RNN layers for each hierarchy level
        for (size_t level = 0; level < numLevels; ++level)
        {
            size_t levelHiddenSize = hiddenSize / std::pow(2, numLevels - level - 1);
            levelHiddenSize = std::max(size_t(16), levelHiddenSize); // Minimum size
            
            if (rnnType == 0) // LSTM
            {
                auto lstm = std::make_unique<LSTM<>>(inputSize, levelHiddenSize);
                rnnLayers.push_back(std::move(lstm));
            }
            else // GRU
            {
                auto gru = std::make_unique<GRU<>>(inputSize, levelHiddenSize);
                rnnLayers.push_back(std::move(gru));
            }
            
            // Add dropout for regularization
            if (dropoutRate > 0.0)
            {
                auto dropout = std::make_unique<Dropout<>>(dropoutRate);
                dropoutLayers.push_back(std::move(dropout));
            }
            
            // Level-specific parameters
            levelFactors.push_back(std::pow(2, level)); // Temporal scaling factor
            levelHiddenSizes.push_back(levelHiddenSize);
            
            std::cout << "Level " << level << ": hidden size=" << levelHiddenSize 
                      << ", factor=" << levelFactors[level] << std::endl;
        }
        
        // Initialize residual connections
        residualWeights.set_size(hiddenSize, inputSize);
        residualWeights.ones(); // Identity initialization
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input shape: (inputSize * sequenceLength, batchSize)
        const size_t batchSize = input.n_cols;
        const size_t seqLen = input.n_rows / inputSize;
        
        output.set_size(hiddenSize * seqLen, batchSize);
        output.zeros();
        
        // Process each hierarchy level
        for (size_t level = 0; level < numLevels; ++level)
        {
            ProcessLevel(level, input, output, batchSize, seqLen);
        }
        
        // Store for backward pass and residual connections
        previousInput = input;
        previousOutput = output;
    }

    void ProcessLevel(size_t level, const MatType& input, MatType& output,
                     size_t batchSize, size_t seqLen)
    {
        size_t levelFactor = levelFactors[level];
        size_t levelHiddenSize = levelHiddenSizes[level];
        
        // Downsample input for this level
        MatType levelInput;
        DownsampleInput(input, levelInput, levelFactor, batchSize, seqLen);
        
        // Process through RNN layer
        MatType levelOutput;
        rnnLayers[level]->Forward(levelInput, levelOutput);
        
        // Apply dropout
        if (dropoutRate > 0.0)
        {
            dropoutLayers[level]->Forward(levelOutput, levelOutput);
        }
        
        // Upsample and add to output
        MatType upsampledOutput;
        UpsampleOutput(levelOutput, upsampledOutput, levelFactor, batchSize, seqLen);
        
        // Add residual connection from input if same dimension
        if (level == 0 && input.n_rows == output.n_rows)
        {
            MatType residual = residualWeights * input;
            output += residual + upsampledOutput;
        }
        else
        {
            output += upsampledOutput;
        }
    }

    void DownsampleInput(const MatType& input, MatType& downsampled,
                        size_t factor, size_t batchSize, size_t seqLen)
    {
        size_t downsampledLen = seqLen / factor;
        downsampled.set_size(inputSize * downsampledLen, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t t = 0; t < downsampledLen; ++t)
            {
                // Average pooling over factor steps
                MatType window = input.rows(t * factor * inputSize, 
                                          (t * factor + factor) * inputSize - 1, b);
                MatType averaged = mean(window, 1);
                downsampled.rows(t * inputSize, (t + 1) * inputSize - 1, b) = averaged;
            }
        }
    }

    void UpsampleOutput(const MatType& input, MatType& upsampled,
                       size_t factor, size_t batchSize, size_t seqLen)
    {
        size_t inputLen = input.n_rows / hiddenSize;
        upsampled.set_size(hiddenSize * seqLen, batchSize);
        upsampled.zeros();
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t t = 0; t < inputLen; ++t)
            {
                // Repeat each step factor times
                MatType stepVector = input.rows(t * hiddenSize, (t + 1) * hiddenSize - 1, b);
                for (size_t f = 0; f < factor; ++f)
                {
                    size_t outputPos = t * factor + f;
                    if (outputPos < seqLen)
                    {
                        upsampled.rows(outputPos * hiddenSize, 
                                     (outputPos + 1) * hiddenSize - 1, b) = stepVector;
                    }
                }
            }
        }
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // Simplified backward pass - in practice, implement full gradient computation
        g = gy;
        
        // Add gradient from residual connection
        if (previousInput.n_rows == gy.n_rows)
        {
            g += residualWeights.t() * gy;
        }
    }

    virtual HierarchicalRNN* Clone() const
    {
        return new HierarchicalRNN(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inputSize));
        ar(CEREAL_NVP(hiddenSize));
        ar(CEREAL_NVP(numLevels));
        ar(CEREAL_NVP(rnnType));
        ar(CEREAL_NVP(dropoutRate));
        ar(CEREAL_NVP(rnnLayers));
        ar(CEREAL_NVP(dropoutLayers));
        ar(CEREAL_NVP(residualWeights));
    }

private:
    size_t inputSize;
    size_t hiddenSize;
    size_t numLevels;
    size_t rnnType;
    double dropoutRate;
    
    std::vector<std::unique_ptr<Layer<MatType>>> rnnLayers;
    std::vector<std::unique_ptr<Dropout<>>> dropoutLayers;
    std::vector<size_t> levelFactors;
    std::vector<size_t> levelHiddenSizes;
    MatType residualWeights;
    MatType previousInput;
    MatType previousOutput;
};

/**
 * Multi-Head Attention Layer for Sequence Modeling
 */
template<typename MatType = arma::mat>
class MultiHeadAttention : public Layer<MatType>
{
public:
    MultiHeadAttention(const size_t hiddenSize,
                       const size_t numHeads = 8,
                       const size_t keySize = 64) :
        Layer<MatType>(),
        hiddenSize(hiddenSize),
        numHeads(numHeads),
        keySize(keySize)
    {
        InitializeAttentionWeights();
    }

    void InitializeAttentionWeights()
    {
        // Initialize query, key, value projections
        WQ.set_size(hiddenSize, numHeads * keySize);
        WK.set_size(hiddenSize, numHeads * keySize);
        WV.set_size(hiddenSize, numHeads * keySize);
        WO.set_size(numHeads * keySize, hiddenSize);
        
        // He initialization
        double stddev = std::sqrt(2.0 / hiddenSize);
        WQ.randn(); WQ *= stddev;
        WK.randn(); WK *= stddev;
        WV.randn(); WV *= stddev;
        WO.randn(); WO *= stddev;
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input shape: (hiddenSize * sequenceLength, batchSize)
        const size_t batchSize = input.n_cols;
        const size_t seqLen = input.n_rows / hiddenSize;
        
        output.set_size(hiddenSize * seqLen, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType sequence = input.col(b);
            sequence.reshape(hiddenSize, seqLen);
            
            // Compute query, key, value projections
            MatType Q = WQ.t() * sequence;
            MatType K = WK.t() * sequence;
            MatType V = WV.t() * sequence;
            
            // Reshape for multi-head attention
            Q.reshape(keySize, numHeads * seqLen);
            K.reshape(keySize, numHeads * seqLen);
            V.reshape(keySize, numHeads * seqLen);
            
            // Compute attention scores
            MatType attentionScores = Q.t() * K / std::sqrt(keySize);
            attentionScores.reshape(seqLen, numHeads * seqLen);
            
            // Apply softmax
            for (size_t i = 0; i < attentionScores.n_cols; ++i)
            {
                attentionScores.col(i) = arma::exp(attentionScores.col(i));
                attentionScores.col(i) /= arma::sum(attentionScores.col(i));
            }
            
            // Apply attention to values
            MatType attentionOutput = attentionScores * V.t();
            attentionOutput.reshape(hiddenSize, seqLen);
            
            // Final projection
            MatType finalOutput = WO * attentionOutput;
            output.col(b) = arma::vectorise(finalOutput);
        }
        
        // Store for residual connection
        attentionOutput = output;
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // Simplified backward pass
        g = gy;
    }

    virtual MultiHeadAttention* Clone() const
    {
        return new MultiHeadAttention(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(hiddenSize));
        ar(CEREAL_NVP(numHeads));
        ar(CEREAL_NVP(keySize));
        ar(CEREAL_NVP(WQ));
        ar(CEREAL_NVP(WK));
        ar(CEREAL_NVP(WV));
        ar(CEREAL_NVP(WO));
    }

private:
    size_t hiddenSize;
    size_t numHeads;
    size_t keySize;
    
    MatType WQ, WK, WV, WO;
    MatType attentionOutput;
};

/**
 * Complete Hierarchical RNN with Attention and Residual Connections
 */
class HierarchicalAttentionRNN
{
public:
    HierarchicalAttentionRNN(const size_t inputSize,
                             const size_t hiddenSize,
                             const size_t outputSize,
                             const size_t numLevels = 3,
                             const size_t numHeads = 8,
                             const double learningRate = 0.001) :
        inputSize(inputSize),
        hiddenSize(hiddenSize),
        outputSize(outputSize),
        numLevels(numLevels),
        numHeads(numHeads),
        learningRate(learningRate)
    {
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        network = std::make_unique<FFN<MeanSquaredError<>, HeInitialization>>();
        
        std::cout << "Building Hierarchical RNN with Attention..." << std::endl;
        
        // Input projection
        network->Add<Linear<>>(inputSize, hiddenSize);
        
        // Hierarchical RNN layers
        network->Add<HierarchicalRNN<>>(hiddenSize, hiddenSize, numLevels, 0, 0.2);
        
        // Residual connection
        network->Add<Linear<>>(hiddenSize, hiddenSize); // Residual projection
        
        // Multi-head attention
        network->Add<MultiHeadAttention<>>(hiddenSize, numHeads);
        
        // Another residual connection
        network->Add<Linear<>>(hiddenSize, hiddenSize);
        
        // Output layers
        network->Add<LSTM<>>(hiddenSize, hiddenSize);
        network->Add<ReLULayer<>>();
        network->Add<Dropout<>>(0.3);
        network->Add<Linear<>>(hiddenSize, outputSize);
        
        std::cout << "Network architecture:" << std::endl;
        std::cout << "  Input size: " << inputSize << std::endl;
        std::cout << "  Hidden size: " << hiddenSize << std::endl;
        std::cout << "  Output size: " << outputSize << std::endl;
        std::cout << "  Hierarchy levels: " << numLevels << std::endl;
        std::cout << "  Attention heads: " << numHeads << std::endl;
    }

    /**
     * Train the hierarchical RNN
     */
    void Train(const mat& sequences, const mat& targets,
               const size_t sequenceLength,
               const size_t epochs = 100,
               const size_t batchSize = 32)
    {
        std::cout << "\nStarting training..." << std::endl;
        std::cout << "Sequences: " << sequences.n_cols << std::endl;
        std::cout << "Sequence length: " << sequenceLength << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        
        ens::Adam optimizer(learningRate, batchSize, 0.9, 0.999, 1e-8);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            double epochLoss = 0.0;
            size_t numBatches = 0;
            
            for (size_t i = 0; i < sequences.n_cols; i += batchSize)
            {
                size_t currentBatchSize = std::min(batchSize, sequences.n_cols - i);
                
                // Reshape sequences for RNN processing
                mat batchSequences = ReshapeSequences(
                    sequences.cols(i, i + currentBatchSize - 1), sequenceLength);
                mat batchTargets = ReshapeSequences(
                    targets.cols(i, i + currentBatchSize - 1), sequenceLength);
                
                double loss = network->Evaluate(batchSequences, batchTargets);
                network->Gradient(batchSequences, batchTargets);
                optimizer.Update(network->Parameters().parameters(),
                               network->Parameters().gradient());
                
                epochLoss += loss;
                numBatches++;
                
                if (numBatches % 10 == 0)
                {
                    std::cout << "Epoch " << epoch + 1 << ", Batch " << numBatches 
                              << ", Loss: " << loss << std::endl;
                }
            }
            
            double avgLoss = epochLoss / numBatches;
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Average Loss: " << avgLoss << std::endl;
        }
    }

    /**
     * Predict next sequence steps
     */
    mat Predict(const mat& inputSequence, const size_t predictionSteps)
    {
        mat currentInput = inputSequence;
        mat predictions(outputSize, predictionSteps);
        
        for (size_t step = 0; step < predictionSteps; ++step)
        {
            mat stepPrediction;
            network->Predict(currentInput, stepPrediction);
            
            // Get the last time step prediction
            mat lastStep = stepPrediction.tail_rows(outputSize);
            predictions.col(step) = lastStep;
            
            // Update input for next prediction (autoregressive)
            if (step < predictionSteps - 1)
            {
                // Shift window and append prediction
                currentInput = UpdateSequenceWindow(currentInput, lastStep);
            }
        }
        
        return predictions;
    }

    /**
     * Evaluate model on test sequences
     */
    double Evaluate(const mat& testSequences, const mat& testTargets,
                   const size_t sequenceLength)
    {
        double totalError = 0.0;
        size_t numSamples = 0;
        
        for (size_t i = 0; i < testSequences.n_cols; ++i)
        {
            mat sequence = ReshapeSequences(testSequences.col(i), sequenceLength);
            mat target = ReshapeSequences(testTargets.col(i), sequenceLength);
            
            mat prediction;
            network->Predict(sequence, prediction);
            
            double error = arma::accu(arma::square(prediction - target));
            totalError += error;
            numSamples++;
        }
        
        return totalError / numSamples;
    }

    /**
     * Reshape flat sequences to RNN format
     */
    mat ReshapeSequences(const mat& flatSequences, const size_t sequenceLength)
    {
        const size_t batchSize = flatSequences.n_cols;
        const size_t featureSize = flatSequences.n_rows / sequenceLength;
        
        mat reshaped(featureSize * sequenceLength, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            for (size_t t = 0; t < sequenceLength; ++t)
            {
                reshaped.rows(t * featureSize, (t + 1) * featureSize - 1, b) =
                    flatSequences.rows(t * featureSize, (t + 1) * featureSize - 1, b);
            }
        }
        
        return reshaped;
    }

    /**
     * Update sequence window for autoregressive prediction
     */
    mat UpdateSequenceWindow(const mat& currentWindow, const mat& newPrediction)
    {
        const size_t seqLen = currentWindow.n_rows / inputSize;
        mat updatedWindow = currentWindow;
        
        // Shift window left
        for (size_t t = 0; t < seqLen - 1; ++t)
        {
            updatedWindow.rows(t * inputSize, (t + 1) * inputSize - 1) =
                currentWindow.rows((t + 1) * inputSize, (t + 2) * inputSize - 1);
        }
        
        // Add new prediction at the end
        updatedWindow.rows((seqLen - 1) * inputSize, seqLen * inputSize - 1) =
            newPrediction;
        
        return updatedWindow;
    }

    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "hierarchical_rnn", *network);
        std::cout << "Model saved to: " << filename << std::endl;
    }

    void LoadModel(const std::string& filename)
    {
        data::Load(filename, "hierarchical_rnn", *network);
        std::cout << "Model loaded from: " << filename << std::endl;
    }

private:
    size_t inputSize;
    size_t hiddenSize;
    size_t outputSize;
    size_t numLevels;
    size_t numHeads;
    double learningRate;
    
    std::unique_ptr<FFN<MeanSquaredError<>, HeInitialization>> network;
};

/**
 * Example: Time Series Prediction with Hierarchical RNN
 */
class TimeSeriesPredictor
{
public:
    TimeSeriesPredictor(const size_t inputDim = 1,
                        const size_t outputDim = 1,
                        const size_t hiddenSize = 128,
                        const size_t sequenceLength = 50) :
        inputDim(inputDim),
        outputDim(outputDim),
        hiddenSize(hiddenSize),
        sequenceLength(sequenceLength),
        model(inputDim, hiddenSize, outputDim, 3, 8, 0.001)
    {}

    /**
     * Generate synthetic time series data
     */
    void GenerateSyntheticData(size_t numSequences = 1000)
    {
        std::cout << "Generating synthetic time series data..." << std::endl;
        
        sequences.set_size(inputDim * sequenceLength, numSequences);
        targets.set_size(outputDim * sequenceLength, numSequences);
        
        for (size_t i = 0; i < numSequences; ++i)
        {
            // Generate multi-scale time series
            mat sequence = GenerateMultiScaleSequence(sequenceLength);
            mat target = GenerateTarget(sequence);
            
            sequences.col(i) = arma::vectorise(sequence);
            targets.col(i) = arma::vectorise(target);
        }
        
        std::cout << "Generated " << numSequences << " sequences" << std::endl;
    }

    mat GenerateMultiScaleSequence(size_t length)
    {
        mat sequence(inputDim, length);
        
        // Multiple frequency components
        for (size_t t = 0; t < length; ++t)
        {
            double highFreq = std::sin(2 * M_PI * t / 5.0);    // High frequency
            double midFreq = std::sin(2 * M_PI * t / 20.0);    // Medium frequency  
            double lowFreq = std::sin(2 * M_PI * t / 100.0);   // Low frequency
            
            sequence(0, t) = 0.5 * highFreq + 0.3 * midFreq + 0.2 * lowFreq +
                           0.1 * arma::randn(); // Add noise
        }
        
        return sequence;
    }

    mat GenerateTarget(const mat& sequence)
    {
        // Predict next step (autoregressive target)
        mat target(outputDim, sequence.n_cols);
        target.zeros();
        
        for (size_t t = 0; t < sequence.n_cols - 1; ++t)
        {
            target.col(t) = sequence.col(t + 1);
        }
        
        // Last target is prediction of future beyond sequence
        target.col(sequence.n_cols - 1) = 0.5 * sequence.col(sequence.n_cols - 1) +
                                        0.3 * sequence.col(sequence.n_cols - 2);
        
        return target;
    }

    void TrainTestSplit(double trainRatio = 0.8)
    {
        size_t totalSequences = sequences.n_cols;
        size_t trainSize = totalSequences * trainRatio;
        
        trainSequences = sequences.cols(0, trainSize - 1);
        trainTargets = targets.cols(0, trainSize - 1);
        testSequences = sequences.cols(trainSize, totalSequences - 1);
        testTargets = targets.cols(trainSize, totalSequences - 1);
        
        std::cout << "Train sequences: " << trainSequences.n_cols << std::endl;
        std::cout << "Test sequences: " << testSequences.n_cols << std::endl;
    }

    void Train(size_t epochs = 100)
    {
        model.Train(trainSequences, trainTargets, sequenceLength, epochs, 32);
        
        // Evaluate on test set
        double testError = model.Evaluate(testSequences, testTargets, sequenceLength);
        std::cout << "Test MSE: " << testError << std::endl;
    }

    void PredictExample()
    {
        // Use first test sequence for prediction example
        mat testSequence = testSequences.col(0);
        mat prediction = model.Predict(testSequence, 10);
        
        std::cout << "Prediction example:" << std::endl;
        std::cout << "Input sequence length: " << sequenceLength << std::endl;
        std::cout << "Predicted next 10 steps shape: " << prediction.n_rows 
                  << " x " << prediction.n_cols << std::endl;
    }

private:
    size_t inputDim;
    size_t outputDim;
    size_t hiddenSize;
    size_t sequenceLength;
    
    mat sequences, targets;
    mat trainSequences, trainTargets;
    mat testSequences, testTargets;
    
    HierarchicalAttentionRNN model;
};

/**
 * Main example usage
 */
int main()
{
    std::cout << "Hierarchical RNN with Attention and Residual Connections" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Create time series predictor
    TimeSeriesPredictor predictor(1, 1, 128, 50);
    
    // Generate and split data
    predictor.GenerateSyntheticData(1000);
    predictor.TrainTestSplit(0.8);
    
    // Train the model
    predictor.Train(50);
    
    // Make predictions
    predictor.PredictExample();
    
    return 0;
}