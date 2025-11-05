#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace ens;

// Self-Attention Time Series Forecaster class
class SelfAttentionForecaster
{
public:
    SelfAttentionForecaster(size_t sequenceLength, 
                          size_t inputDim, 
                          size_t hiddenDim,
                          size_t numHeads,
                          size_t numLayers,
                          size_t forecastHorizon)
        : sequenceLength(sequenceLength)
        , inputDim(inputDim)
        , hiddenDim(hiddenDim)
        , numHeads(numHeads)
        , numLayers(numLayers)
        , forecastHorizon(forecastHorizon)
    {
        BuildModel();
    }

    // Build the self-attention based model
    void BuildModel()
    {
        // Input layer
        model.Add<IdentityLayer<> >();
        
        // Positional encoding (learnable)
        model.Add<Linear<> >(inputDim * sequenceLength, hiddenDim);
        model.Add<ReLULayer<> >();
        
        // Multiple self-attention layers
        for (size_t i = 0; i < numLayers; ++i)
        {
            // Self-attention mechanism
            model.Add<MultiheadAttention<> >(hiddenDim, numHeads);
            model.Add<LayerNorm<> >(hiddenDim);
            model.Add<ReLULayer<> >();
            
            // Feed-forward network
            model.Add<Linear<> >(hiddenDim, hiddenDim * 4);
            model.Add<ReLULayer<> >();
            model.Add<Linear<> >(hiddenDim * 4, hiddenDim);
            model.Add<LayerNorm<> >(hiddenDim);
            model.Add<ReLULayer<> >();
        }
        
        // Global average pooling over sequence dimension
        model.Add<MeanPooling<> >(sequenceLength, 1);
        
        // Forecasting head
        model.Add<Linear<> >(hiddenDim, hiddenDim / 2);
        model.Add<ReLULayer<> >();
        model.Add<Dropout<> >(0.1);
        model.Add<Linear<> >(hiddenDim / 2, forecastHorizon);
        
        // Initialize model parameters
        model.ResetParameters();
    }

    // Train the model
    void Train(const mat& trainData, const mat& trainLabels,
               const mat& validationData, const mat& validationLabels,
               size_t epochs = 100,
               double learningRate = 0.001)
    {
        // Create Adam optimizer
        Adam optimizer(learningRate, 32, 0.9, 0.999, 1e-8, epochs * trainData.n_cols, 1e-8, true);
        
        // Training callback to print progress
        auto printLoss = [&](const arma::mat& /* params */,
                            const size_t epoch,
                            const double loss)
        {
            if (epoch % 10 == 0)
            {
                double valLoss = Evaluate(validationData, validationLabels);
                std::cout << "Epoch: " << epoch << " - Training Loss: " << loss
                         << " - Validation Loss: " << valLoss << std::endl;
            }
        };
        
        // Train the model
        model.Train(trainData, trainLabels, optimizer, printLoss);
    }

    // Evaluate model on test data
    double Evaluate(const mat& testData, const mat& testLabels)
    {
        mat predictions;
        model.Predict(testData, predictions);
        
        // Calculate Mean Squared Error
        double mse = mean(square(predictions - testLabels));
        return mse;
    }

    // Generate forecasts
    mat Predict(const mat& inputData)
    {
        mat predictions;
        model.Predict(inputData, predictions);
        return predictions;
    }

    // Save model to file
    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "model", model);
    }

    // Load model from file
    void LoadModel(const std::string& filename)
    {
        data::Load(filename, "model", model);
    }

private:
    FFN<MeanSquaredError<>, HeInitialization> model;
    size_t sequenceLength;
    size_t inputDim;
    size_t hiddenDim;
    size_t numHeads;
    size_t numLayers;
    size_t forecastHorizon;
};

// Data preprocessing functions
class TimeSeriesPreprocessor
{
public:
    // Create sliding window dataset for time series forecasting
    static void CreateSlidingWindowDataset(const vec& timeSeries,
                                         size_t sequenceLength,
                                         size_t forecastHorizon,
                                         mat& features,
                                         mat& labels)
    {
        size_t totalSamples = timeSeries.n_elem - sequenceLength - forecastHorizon + 1;
        
        features.set_size(sequenceLength, totalSamples);
        labels.set_size(forecastHorizon, totalSamples);
        
        for (size_t i = 0; i < totalSamples; ++i)
        {
            // Input sequence
            features.col(i) = timeSeries.subvec(i, i + sequenceLength - 1);
            
            // Target values (next forecastHorizon steps)
            labels.col(i) = timeSeries.subvec(i + sequenceLength, 
                                            i + sequenceLength + forecastHorizon - 1);
        }
    }

    // Normalize data using min-max scaling
    static void NormalizeData(mat& data, rowvec& minVals, rowvec& maxVals)
    {
        minVals = min(data, 1).t();
        maxVals = max(data, 1).t();
        
        for (size_t i = 0; i < data.n_cols; ++i)
        {
            data.col(i) = (data.col(i) - minVals.t()) / (maxVals.t() - minVals.t() + 1e-8);
        }
    }

    // Denormalize data
    static void DenormalizeData(mat& data, const rowvec& minVals, const rowvec& maxVals)
    {
        for (size_t i = 0; i < data.n_cols; ++i)
        {
            data.col(i) = data.col(i) % (maxVals.t() - minVals.t()) + minVals.t();
        }
    }

    // Split data into train and validation sets
    static void TrainTestSplit(const mat& features, const mat& labels,
                              double testRatio,
                              mat& trainFeatures, mat& trainLabels,
                              mat& testFeatures, mat& testLabels)
    {
        size_t totalSamples = features.n_cols;
        size_t testSize = totalSamples * testRatio;
        size_t trainSize = totalSamples - testSize;
        
        trainFeatures = features.cols(0, trainSize - 1);
        trainLabels = labels.cols(0, trainSize - 1);
        testFeatures = features.cols(trainSize, totalSamples - 1);
        testLabels = labels.cols(trainSize, totalSamples - 1);
    }
};

// Example usage with synthetic data
int main()
{
    // Generate synthetic time series data
    size_t dataLength = 1000;
    vec timeSeries(dataLength);
    
    // Create a synthetic time series with trend and seasonality
    for (size_t i = 0; i < dataLength; ++i)
    {
        double t = static_cast<double>(i);
        timeSeries(i) = 0.1 * t + 5 * sin(2 * M_PI * t / 50) + 2 * sin(2 * M_PI * t / 25);
    }
    
    // Add some noise
    timeSeries += 0.5 * randn<vec>(dataLength);
    
    // Parameters
    size_t sequenceLength = 50;
    size_t forecastHorizon = 10;
    size_t hiddenDim = 64;
    size_t numHeads = 4;
    size_t numLayers = 2;
    
    // Preprocess data
    mat features, labels;
    TimeSeriesPreprocessor::CreateSlidingWindowDataset(timeSeries, 
                                                      sequenceLength, 
                                                      forecastHorizon,
                                                      features, 
                                                      labels);
    
    // Normalize data
    rowvec minVals, maxVals;
    TimeSeriesPreprocessor::NormalizeData(features, minVals, maxVals);
    TimeSeriesPreprocessor::NormalizeData(labels, minVals, maxVals);
    
    // Split data
    mat trainFeatures, trainLabels, testFeatures, testLabels;
    TimeSeriesPreprocessor::TrainTestSplit(features, labels, 0.2,
                                          trainFeatures, trainLabels,
                                          testFeatures, testLabels);
    
    // Create and train the forecaster
    SelfAttentionForecaster forecaster(sequenceLength, 1, hiddenDim, 
                                     numHeads, numLayers, forecastHorizon);
    
    std::cout << "Starting training..." << std::endl;
    forecaster.Train(trainFeatures, trainLabels, testFeatures, testLabels,
                    100, 0.001);
    
    // Evaluate final performance
    double finalLoss = forecaster.Evaluate(testFeatures, testLabels);
    std::cout << "Final Test Loss (MSE): " << finalLoss << std::endl;
    
    // Make predictions
    mat predictions = forecaster.Predict(testFeatures);
    
    // Denormalize predictions and actual values for comparison
    TimeSeriesPreprocessor::DenormalizeData(predictions, minVals, maxVals);
    mat actualValues = testLabels;
    TimeSeriesPreprocessor::DenormalizeData(actualValues, minVals, maxVals);
    
    // Print some example predictions
    std::cout << "\nExample predictions vs actual:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), testFeatures.n_cols); ++i)
    {
        std::cout << "Sample " << i << " - Predicted: " << predictions.col(i).t()
                 << " Actual: " << actualValues.col(i).t() << std::endl;
    }
    
    // Save the trained model
    forecaster.SaveModel("self_attention_forecaster.bin");
    std::cout << "Model saved to self_attention_forecaster.bin" << std::endl;
    
    return 0;
}