#include "CPCTrainer.hpp"
#include <iomanip>

CPCTrainer::CPCTrainer(
    size_t inputDim,
    size_t encoderDim,
    size_t contextDim,
    size_t numPredictions,
    size_t negativeSamples,
    double learningRate
) : cpc(inputDim, encoderDim, contextDim, numPredictions, negativeSamples, learningRate),
    dataset(100, inputDim, 1000),  // sequenceLength=100, numSequences=1000
    inputDim(inputDim) {
}

void CPCTrainer::Train(size_t epochs, size_t sequencesPerEpoch) {
    std::cout << "Starting CPC Training" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "Input Dimension: " << inputDim << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Sequences per epoch: " << sequencesPerEpoch << std::endl;
    
    // Generate training data
    auto sequences = dataset.GenerateMultimodalData();
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double epochLoss = 0.0;
        size_t processedSequences = 0;
        
        for (size_t i = 0; i < sequencesPerEpoch && i < sequences.size(); ++i) {
            double loss = cpc.Train(sequences[i]);
            epochLoss += loss;
            processedSequences++;
        }
        
        epochLoss /= processedSequences;
        trainingLoss.push_back(epochLoss);
        
        PrintTrainingProgress(epoch, epochs, epochLoss);
        
        // Save model periodically
        if (epoch % 10 == 0) {
            cpc.Save("cpc_model_epoch_" + std::to_string(epoch));
        }
    }
    
    // Save final model
    cpc.Save("cpc_model_final");
    
    std::cout << "Training completed!" << std::endl;
}

void CPCTrainer::Evaluate() {
    std::cout << "\nEvaluating CPC Model" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Generate test sequences
    auto testSequences = dataset.GenerateMultimodalData();
    
    // Test encoding
    if (!testSequences.empty()) {
        arma::mat embeddings = cpc.Encode(testSequences[0].t());
        std::cout << "Embedding shape: " << embeddings.n_rows << " x " << embeddings.n_cols << std::endl;
        
        // Test context extraction
        arma::mat context = cpc.GetContext(testSequences[0]);
        std::cout << "Context shape: " << context.n_rows << " x " << context.n_cols << std::endl;
    }
    
    // Print final loss
    if (!trainingLoss.empty()) {
        std::cout << "Final training loss: " << trainingLoss.back() << std::endl;
    }
}

void CPCTrainer::VisualizeEmbeddings() {
    // This would typically use a visualization library
    // For now, we'll just print some statistics
    
    auto sequences = dataset.GenerateMultimodalData();
    arma::mat allEmbeddings;
    
    for (size_t i = 0; i < std::min(size_t(10), sequences.size()); ++i) {
        arma::mat embeddings = cpc.Encode(sequences[i].t());
        if (i == 0) {
            allEmbeddings = embeddings;
        } else {
            allEmbeddings = arma::join_rows(allEmbeddings, embeddings);
        }
    }
    
    std::cout << "\nEmbedding Statistics:" << std::endl;
    std::cout << "Mean: " << arma::mean(arma::vectorise(allEmbeddings)) << std::endl;
    std::cout << "Std:  " << arma::stddev(arma::vectorise(allEmbeddings)) << std::endl;
    std::cout << "Min:  " << arma::min(arma::vectorise(allEmbeddings)) << std::endl;
    std::cout << "Max:  " << arma::max(arma::vectorise(allEmbeddings)) << std::endl;
}

void CPCTrainer::PrintTrainingProgress(size_t epoch, size_t totalEpochs, double loss) {
    double progress = 100.0 * (epoch + 1) / totalEpochs;
    
    std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << totalEpochs
              << " [" << std::setw(5) << std::fixed << std::setprecision(1) << progress << "%]"
              << " - Loss: " << std::setw(10) << std::scientific << loss;
    
    // Show loss trend
    if (epoch >= 1) {
        double lossChange = trainingLoss[epoch - 1] - loss;
        std::cout << " (Δ: " << std::setw(8) << std::scientific << lossChange << ")";
    }
    
    std::cout << std::endl;
}