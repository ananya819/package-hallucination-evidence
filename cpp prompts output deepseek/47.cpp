#include "MetaTaskGenerator.hpp"
#include <cmath>

namespace meta_learning {

MetaTaskGenerator::MetaTaskGenerator(size_t inputDim,
                                   size_t totalClasses,
                                   size_t numSupportPerClass,
                                   size_t numQueryPerClass)
    : inputDim(inputDim), totalClasses(totalClasses),
      numSupportPerClass(numSupportPerClass), 
      numQueryPerClass(numQueryPerClass),
      rng(std::random_device{}()) {
}

std::vector<Task> MetaTaskGenerator::GenerateTasks(size_t numTasks, 
                                                 size_t numWays,
                                                 const std::string& datasetType) {
    if (datasetType == "omniglot") {
        return GenerateOmniglotTasks(numTasks, numWays);
    } else if (datasetType == "miniimagenet") {
        return GenerateMiniImagenetTasks(numTasks, numWays);
    } else {
        return GenerateOmniglotTasks(numTasks, numWays);
    }
}

std::vector<Task> MetaTaskGenerator::GenerateOmniglotTasks(size_t numTasks, size_t numWays) {
    std::vector<Task> tasks;
    tasks.reserve(numTasks);
    
    std::uniform_real_distribution<double> featureDist(-1.0, 1.0);
    std::uniform_int_distribution<size_t> classDist(0, totalClasses - 1);
    
    for (size_t taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
        Task task;
        task.numClasses = numWays;
        task.numSupportPerClass = numSupportPerClass;
        task.numQueryPerClass = numQueryPerClass;
        
        // Sample random classes for this task
        std::vector<size_t> selectedClasses = SampleRandomClasses(numWays);
        
        size_t totalSupport = numWays * numSupportPerClass;
        size_t totalQuery = numWays * numQueryPerClass;
        
        task.supportFeatures.set_size(inputDim, totalSupport);
        task.supportLabels.set_size(numWays, totalSupport);
        task.queryFeatures.set_size(inputDim, totalQuery);
        task.queryLabels.set_size(numWays, totalQuery);
        
        // Generate support set
        size_t supportIdx = 0;
        for (size_t classIdx = 0; classIdx < numWays; ++classIdx) {
            size_t classLabel = selectedClasses[classIdx];
            
            for (size_t i = 0; i < numSupportPerClass; ++i) {
                // Generate random features for this class
                for (size_t d = 0; d < inputDim; ++d) {
                    task.supportFeatures(d, supportIdx) = featureDist(rng) + classLabel * 0.1;
                }
                
                // One-hot encoding
                task.supportLabels.col(supportIdx).zeros();
                task.supportLabels(classIdx, supportIdx) = 1.0;
                
                supportIdx++;
            }
        }
        
        // Generate query set
        size_t queryIdx = 0;
        for (size_t classIdx = 0; classIdx < numWays; ++classIdx) {
            size_t classLabel = selectedClasses[classIdx];
            
            for (size_t i = 0; i < numQueryPerClass; ++i) {
                // Generate random features for this class
                for (size_t d = 0; d < inputDim; ++d) {
                    task.queryFeatures(d, queryIdx) = featureDist(rng) + classLabel * 0.1;
                }
                
                // One-hot encoding
                task.queryLabels.col(queryIdx).zeros();
                task.queryLabels(classIdx, queryIdx) = 1.0;
                
                queryIdx++;
            }
        }
        
        tasks.push_back(task);
    }
    
    return tasks;
}

std::vector<Task> MetaTaskGenerator::GenerateSineWaveTasks(size_t numTasks,
                                                         size_t numSupport,
                                                         size_t numQuery) {
    std::vector<Task> tasks;
    tasks.reserve(numTasks);
    
    std::uniform_real_distribution<double> ampDist(0.1, 5.0);
    std::uniform_real_distribution<double> phaseDist(0.0, M_PI);
    std::uniform_real_distribution<double> xDist(-5.0, 5.0);
    
    for (size_t taskIdx = 0; taskIdx < numTasks; ++taskIdx) {
        Task task;
        task.numClasses = 1; // Regression task
        task.numSupportPerClass = numSupport;
        task.numQueryPerClass = numQuery;
        
        // Sample task parameters
        double amplitude = ampDist(rng);
        double phase = phaseDist(rng);
        
        // Generate support set
        task.supportFeatures.set_size(1, numSupport);
        task.supportLabels.set_size(1, numSupport);
        
        for (size_t i = 0; i < numSupport; ++i) {
            double x = xDist(rng);
            double y = amplitude * std::sin(x + phase);
            
            task.supportFeatures(0, i) = x;
            task.supportLabels(0, i) = y;
        }
        
        // Generate query set
        task.queryFeatures.set_size(1, numQuery);
        task.queryLabels.set_size(1, numQuery);
        
        for (size_t i = 0; i < numQuery; ++i) {
            double x = xDist(rng);
            double y = amplitude * std::sin(x + phase);
            
            task.queryFeatures(0, i) = x;
            task.queryLabels(0, i) = y;
        }
        
        tasks.push_back(task);
    }
    
    return tasks;
}

std::vector<size_t> MetaTaskGenerator::SampleRandomClasses(size_t numWays) {
    std::vector<size_t> allClasses(totalClasses);
    for (size_t i = 0; i < totalClasses; ++i) {
        allClasses[i] = i;
    }
    
    // Shuffle and select first numWays classes
    std::shuffle(allClasses.begin(), allClasses.end(), rng);
    allClasses.resize(numWays);
    
    return allClasses;
}

} // namespace meta_learning and 
//this is different file for this prompt
#include "PrototypicalNetworks.hpp"
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>

namespace meta_learning {

PrototypicalNetworks::PrototypicalNetworks(size_t inputDim,
                                         size_t embeddingDim,
                                         double learningRate)
    : inputDim(inputDim), embeddingDim(embeddingDim), learningRate(learningRate) {
    
    embeddingNetwork = CreateBaseNetwork();
    optimizer = std::make_unique<Adam>(learningRate, 32, 0.9, 0.999, 1e-8);
}

FFN<MeanSquaredError<>, GaussianInitialization> PrototypicalNetworks::CreateBaseNetwork() {
    FFN<MeanSquaredError<>, GaussianInitialization> network;
    
    // Embedding network architecture
    network.Add<Linear>(128);
    network.Add<ReLU>();
    network.Add<Linear>(256);
    network.Add<ReLU>();
    network.Add<Linear>(embeddingDim);  // Embedding dimension
    
    return network;
}

double PrototypicalNetworks::MetaTrain(const std::vector<Task>& tasks, size_t metaBatchSize) {
    double totalLoss = 0.0;
    size_t processedTasks = 0;
    
    for (const auto& task : tasks) {
        // Compute support embeddings
        arma::mat supportEmbeddings = ComputeEmbeddings(task.supportFeatures);
        
        // Compute prototypes
        arma::mat prototypes = ComputePrototypes(supportEmbeddings, 
                                               task.supportLabels, 
                                               task.numClasses);
        
        // Compute query embeddings
        arma::mat queryEmbeddings = ComputeEmbeddings(task.queryFeatures);
        
        // Compute distances
        arma::mat distances = ComputeDistances(queryEmbeddings, prototypes);
        
        // Compute loss
        double loss = ComputePrototypicalLoss(distances, task.queryLabels, task.numClasses);
        totalLoss += loss;
        processedTasks++;
        
        metaTrainLoss.push_back(loss);
        
        // Training step would update embedding network here
        // This is simplified for demonstration
    }
    
    return totalLoss / processedTasks;
}

double PrototypicalNetworks::MetaTest(const Task& task) {
    // Compute support embeddings
    arma::mat supportEmbeddings = ComputeEmbeddings(task.supportFeatures);
    
    // Compute prototypes
    arma::mat prototypes = ComputePrototypes(supportEmbeddings, 
                                           task.supportLabels, 
                                           task.numClasses);
    
    // Compute query embeddings
    arma::mat queryEmbeddings = ComputeEmbeddings(task.queryFeatures);
    
    // Compute distances
    arma::mat distances = ComputeDistances(queryEmbeddings, prototypes);
    
    // Compute loss
    double loss = ComputePrototypicalLoss(distances, task.queryLabels, task.numClasses);
    metaTestLoss.push_back(loss);
    
    return loss;
}

void PrototypicalNetworks::AdaptToTask(const Task& task, size_t adaptationSteps) {
    // Prototypical networks don't need adaptation - they're non-parametric
    // This method is kept for interface consistency
}

arma::mat PrototypicalNetworks::ComputeEmbeddings(const arma::mat& features) {
    arma::mat embeddings;
    embeddingNetwork.Predict(features, embeddings);
    return embeddings;
}

arma::mat PrototypicalNetworks::ComputePrototypes(const arma::mat& supportEmbeddings,
                                                 const arma::mat& supportLabels,
                                                 size_t numClasses) {
    arma::mat prototypes(embeddingDim, numClasses, arma::fill::zeros);
    arma::vec classCounts(numClasses, arma::fill::zeros);
    
    // Sum embeddings for each class
    for (size_t i = 0; i < supportEmbeddings.n_cols; ++i) {
        // Find which class this support example belongs to
        for (size_t c = 0; c < numClasses; ++c) {
            if (supportLabels(c, i) > 0.5) { // One-hot encoded
                prototypes.col(c) += supportEmbeddings.col(i);
                classCounts(c)++;
                break;
            }
        }
    }
    
    // Average to get prototypes
    for (size_t c = 0; c < numClasses; ++c) {
        if (classCounts(c) > 0) {
            prototypes.col(c) /= classCounts(c);
        }
    }
    
    return prototypes;
}

arma::mat PrototypicalNetworks::ComputeDistances(const arma::mat& queryEmbeddings,
                                                const arma::mat& prototypes) {
    arma::mat distances(prototypes.n_cols, queryEmbeddings.n_cols);
    
    // Compute Euclidean distances
    for (size_t i = 0; i < queryEmbeddings.n_cols; ++i) {
        for (size_t c = 0; c < prototypes.n_cols; ++c) {
            double dist = arma::norm(queryEmbeddings.col(i) - prototypes.col(c), 2);
            distances(c, i) = dist;
        }
    }
    
    return distances;
}

double PrototypicalNetworks::ComputePrototypicalLoss(const arma::mat& distances,
                                                    const arma::mat& queryLabels,
                                                    size_t numClasses) {
    // Convert distances to probabilities using softmax
    arma::mat logits = -distances; // Negative distances as logits
    arma::mat probabilities = arma::exp(logits);
    
    // Normalize
    for (size_t i = 0; i < probabilities.n_cols; ++i) {
        probabilities.col(i) /= arma::sum(probabilities.col(i));
    }
    
    // Compute negative log likelihood
    double loss = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < queryLabels.n_cols; ++i) {
        for (size_t c = 0; c < numClasses; ++c) {
            if (queryLabels(c, i) > 0.5) { // One-hot encoded
                loss -= std::log(std::max(probabilities(c, i), 1e-8));
                count++;
                break;
            }
        }
    }
    
    return loss / count;
}

void PrototypicalNetworks::Save(const std::string& path) {
    embeddingNetwork.Parameters().save(path + "_protonet_embedding.bin");
}

void PrototypicalNetworks::Load(const std::string& path) {
    arma::mat params;
    if (params.load(path + "_protonet_embedding.bin")) {
        embeddingNetwork.Parameters() = params;
    }
}

} // namespace meta_learning