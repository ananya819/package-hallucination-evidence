#include "AdaptiveNeuralOptimizer.hpp"
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <random>
#include <cmath>

AdaptiveNeuralOptimizer::AdaptiveNeuralOptimizer(size_t parameterDim,
                                               size_t hiddenDim,
                                               double learningRate,
                                               size_t historySize)
    : parameterDim(parameterDim), hiddenDim(hiddenDim),
      learningRate(learningRate), historySize(historySize) {
    
    BuildUpdateNetwork();
    BuildStateEncoder();
    BuildPolicyNetwork();
}

void AdaptiveNeuralOptimizer::BuildUpdateNetwork() {
    // Network that computes parameter updates
    // Input: gradient features, loss history, optimizer state
    size_t inputDim = parameterDim * 3 + hiddenDim + historySize * 2;
    
    updateNetwork.Add<Linear>(512);
    updateNetwork.Add<LeakyReLU>(0.1);
    updateNetwork.Add<Linear>(256);
    updateNetwork.Add<LeakyReLU>(0.1);
    updateNetwork.Add<Linear>(128);
    updateNetwork.Add<LeakyReLU>(0.1);
    updateNetwork.Add<Linear>(parameterDim);  // Output: parameter update
    updateNetwork.Add<Tanh>();  // Bounded updates
}

void AdaptiveNeuralOptimizer::BuildStateEncoder() {
    // Encoder for optimizer state
    size_t inputDim = parameterDim * 2 + historySize * 3;
    
    stateEncoder.Add<Linear>(256);
    stateEncoder.Add<LeakyReLU>(0.1);
    stateEncoder.Add<Linear>(hiddenDim);
    stateEncoder.Add<LeakyReLU>(0.1);
    stateEncoder.Add<Linear>(hiddenDim);  // Encoded state
}

void AdaptiveNeuralOptimizer::BuildPolicyNetwork() {
    // Policy network for RL training
    size_t inputDim = hiddenDim + parameterDim * 2;
    
    policyNetwork.Add<Linear>(256);
    policyNetwork.Add<LeakyReLU>(0.1);
    policyNetwork.Add<Linear>(128);
    policyNetwork.Add<LeakyReLU>(0.1);
    policyNetwork.Add<Linear>(64);
    policyNetwork.Add<LeakyReLU>(0.1);
    policyNetwork.Add<Linear>(1);  // Value estimate
}

arma::mat AdaptiveNeuralOptimizer::ComputeUpdate(const arma::mat& parameters,
                                               const arma::mat& gradients,
                                               double currentLoss,
                                               OptimizerState& state) {
    // Prepare feature vector for the optimizer
    std::vector<OptimizationStep> recentHistory;
    
    // For now, use empty history - in practice you'd maintain this
    arma::mat features = ComputeOptimizerFeatures(parameters, gradients, 
                                                currentLoss, state, recentHistory);
    
    // Compute update using neural network
    arma::mat rawUpdate;
    updateNetwork.Predict(features, rawUpdate);
    
    // Scale update by learning rate
    arma::mat update = learningRate * rawUpdate;
    
    // Update optimizer state
    state.stepCount++;
    
    return update;
}

arma::mat AdaptiveNeuralOptimizer::ComputeOptimizerFeatures(
    const arma::mat& parameters,
    const arma::mat& gradients,
    double loss,
    const OptimizerState& state,
    const std::vector<OptimizationStep>& history) {
    
    size_t totalFeatures = parameterDim * 3 + hiddenDim + historySize * 2;
    arma::mat features(totalFeatures, 1, arma::fill::zeros);
    
    size_t featureIdx = 0;
    
    // Current parameters (normalized)
    arma::mat normParams = arma::normalise(parameters);
    features.submat(featureIdx, 0, featureIdx + parameterDim - 1, 0) = normParams;
    featureIdx += parameterDim;
    
    // Current gradients (normalized)
    arma::mat normGrads = NormalizeGradients(gradients);
    features.submat(featureIdx, 0, featureIdx + parameterDim - 1, 0) = normGrads;
    featureIdx += parameterDim;
    
    // Gradient statistics
    arma::mat gradStats = ComputeGradientStatistics(gradients);
    features.submat(featureIdx, 0, featureIdx + parameterDim - 1, 0) = gradStats;
    featureIdx += parameterDim;
    
    // Optimizer hidden state
    features.submat(featureIdx, 0, featureIdx + hiddenDim - 1, 0) = state.hiddenState;
    featureIdx += hiddenDim;
    
    // Loss history (padded with zeros if not enough history)
    for (size_t i = 0; i < historySize; ++i) {
        double histLoss = 0.0;
        if (i < history.size()) {
            histLoss = history[history.size() - 1 - i].loss;
        }
        features(featureIdx++, 0) = histLoss;
    }
    
    // Step count features
    for (size_t i = 0; i < historySize; ++i) {
        double stepFeature = (state.stepCount + i) / 1000.0;  // Normalized step
        features(featureIdx++, 0) = stepFeature;
    }
    
    return features;
}

double AdaptiveNeuralOptimizer::TrainOptimizer(
    const std::vector<OptimizationStep>& trajectory,
    size_t epochs) {
    
    double totalLoss = 0.0;
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Use policy gradient to update the optimizer
        UpdateWithPolicyGradient(trajectory);
        
        // Compute value loss for monitoring
        double epochLoss = 0.0;
        for (size_t t = 0; t < trajectory.size() - 1; ++t) {
            OptimizerState state;
            arma::mat update = ComputeUpdate(trajectory[t].parameters,
                                           trajectory[t].gradients,
                                           trajectory[t].loss,
                                           state);
            
            double reward = ComputeReward(trajectory[t], trajectory[t + 1], update);
            epochLoss += std::abs(reward);
        }
        
        epochLoss /= (trajectory.size() - 1);
        trainingLoss.push_back(epochLoss);
        totalLoss += epochLoss;
    }
    
    return totalLoss / epochs;
}

double AdaptiveNeuralOptimizer::MetaTrain(
    const std::vector<std::vector<OptimizationStep>>& problems,
    size_t metaBatchSize) {
    
    double totalMetaLoss = 0.0;
    size_t processedBatches = 0;
    
    for (size_t batchStart = 0; batchStart < problems.size(); batchStart += metaBatchSize) {
        size_t batchEnd = std::min(batchStart + metaBatchSize, problems.size());
        
        // Sample meta-batch
        std::vector<std::vector<OptimizationStep>> metaBatch;
        for (size_t i = batchStart; i < batchEnd; ++i) {
            metaBatch.push_back(problems[i]);
        }
        
        // Train on meta-batch
        double batchLoss = 0.0;
        for (const auto& problem : metaBatch) {
            batchLoss += TrainOptimizer(problem, 1);  // One epoch per problem
        }
        
        batchLoss /= metaBatch.size();
        totalMetaLoss += batchLoss;
        processedBatches++;
        
        // Add to replay buffer
        if (replayBuffer.size() >= maxReplaySize) {
            replayBuffer.pop_front();
        }
        replayBuffer.push_back(metaBatch[0]);  // Simplified
    }
    
    return totalMetaLoss / processedBatches;
}

double AdaptiveNeuralOptimizer::ComputeReward(const OptimizationStep& current,
                                            const OptimizationStep& next,
                                            const arma::mat& update) {
    // Reward based on improvement in loss
    double lossImprovement = current.loss - next.loss;
    
    // Penalize large updates for stability
    double updatePenalty = -0.01 * arma::accu(arma::square(update));
    
    // Encourage convergence
    double convergenceBonus = (next.loss < 1e-3) ? 1.0 : 0.0;
    
    return lossImprovement + updatePenalty + convergenceBonus;
}

void AdaptiveNeuralOptimizer::UpdateWithPolicyGradient(
    const std::vector<OptimizationStep>& trajectory) {
    
    // Simplified policy gradient update
    // In practice, you'd use proper advantage estimation
    
    std::vector<double> rewards;
    std::vector<arma::mat> states;
    std::vector<arma::mat> actions;
    
    // Collect trajectory data
    OptimizerState state = InitializeState();
    for (size_t t = 0; t < trajectory.size() - 1; ++t) {
        arma::mat update = ComputeUpdate(trajectory[t].parameters,
                                       trajectory[t].gradients,
                                       trajectory[t].loss,
                                       state);
        
        double reward = ComputeReward(trajectory[t], trajectory[t + 1], update);
        
        rewards.push_back(reward);
        states.push_back(state.hiddenState);
        actions.push_back(update);
    }
    
    // Compute returns (simplified)
    double gamma = 0.99;
    std::vector<double> returns(rewards.size());
    double R = 0.0;
    for (int t = rewards.size() - 1; t >= 0; --t) {
        R = rewards[t] + gamma * R;
        returns[t] = R;
    }
    
    // Normalize returns
    double meanReturn = arma::mean(returns);
    double stdReturn = arma::stddev(returns);
    if (stdReturn > 0) {
        for (size_t t = 0; t < returns.size(); ++t) {
            returns[t] = (returns[t] - meanReturn) / stdReturn;
        }
    }
    
    // Policy gradient update would go here
    // This is simplified - actual implementation would compute gradients
}

AdaptiveNeuralOptimizer::OptimizerState AdaptiveNeuralOptimizer::InitializeState() {
    OptimizerState state;
    state.hiddenState = arma::zeros<arma::mat>(hiddenDim, 1);
    state.cellState = arma::zeros<arma::mat>(hiddenDim, 1);
    state.stepCount = 0;
    return state;
}

arma::mat AdaptiveNeuralOptimizer::NormalizeGradients(const arma::mat& gradients) {
    double gradNorm = arma::norm(gradients, 2);
    if (gradNorm > 1e-8) {
        return gradients / gradNorm;
    }
    return gradients;
}

arma::mat AdaptiveNeuralOptimizer::ComputeGradientStatistics(const arma::mat& gradients) {
    arma::mat stats(parameterDim, 1);
    
    // Compute moving statistics (simplified)
    double meanGrad = arma::mean(gradients);
    double stdGrad = arma::stddev(gradients);
    
    if (stdGrad > 1e-8) {
        stats = (gradients - meanGrad) / stdGrad;
    } else {
        stats.zeros();
    }
    
    return stats;
}

void AdaptiveNeuralOptimizer::Save(const std::string& path) {
    updateNetwork.Parameters().save(path + "_update_network.bin");
    stateEncoder.Parameters().save(path + "_state_encoder.bin");
    policyNetwork.Parameters().save(path + "_policy_network.bin");
}

void AdaptiveNeuralOptimizer::Load(const std::string& path) {
    arma::mat params;
    
    if (params.load(path + "_update_network.bin")) {
        updateNetwork.Parameters() = params;
    }
    if (params.load(path + "_state_encoder.bin")) {
        stateEncoder.Parameters() = params;
    }
    if (params.load(path + "_policy_network.bin")) {
        policyNetwork.Parameters() = params;
    }
}