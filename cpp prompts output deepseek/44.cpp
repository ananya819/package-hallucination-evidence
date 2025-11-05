#include "MARLTrainer.hpp"
#include <iostream>

MARLTrainer::MARLTrainer(int numAgents, int stateSize, int actionSize)
    : env(numAgents, stateSize, actionSize),
      dqn(numAgents, stateSize, actionSize) {
}

void MARLTrainer::train(int numEpisodes, int maxStepsPerEpisode) {
    double epsilon = epsilonStart;
    
    for (int episode = 0; episode < numEpisodes; ++episode) {
        auto state = env.reset();
        double totalReward = 0.0;
        
        for (int step = 0; step < maxStepsPerEpisode; ++step) {
            // Select actions for all agents
            auto actions = dqn.selectAction(state.observations, epsilon);
            MultiAgentEnvironment::Action action{actions};
            
            // Take step in environment
            auto [nextState, reward, done] = env.step(action);
            
            // Store experiences for each agent
            for (int i = 0; i < env.getNumAgents(); ++i) {
                Experience exp;
                exp.state = state.observations.col(i);
                exp.action = actions.col(i);
                exp.reward = reward.rewards(i);
                exp.nextState = nextState.observations.col(i);
                exp.done = done;
                exp.agentId = i;
                
                dqn.storeExperience(exp);
            }
            
            // Train the DQN
            dqn.train();
            
            // Update statistics
            totalReward += arma::mean(reward.rewards);
            state = nextState;
            
            if (done) break;
        }
        
        // Update target networks periodically
        if (episode % 10 == 0) {
            dqn.updateTargetNetworks();
        }
        
        // Decay epsilon
        epsilon = std::max(epsilonEnd, epsilon * epsilonDecay);
        
        // Record statistics
        episodeRewards.push_back(totalReward);
        
        // Calculate running average
        double avgReward = 0.0;
        int window = std::min(100, episode + 1);
        for (int i = std::max(0, episode - window + 1); i <= episode; ++i) {
            avgReward += episodeRewards[i];
        }
        avgReward /= window;
        averageRewards.push_back(avgReward);
        
        // Print progress
        if (episode % 100 == 0) {
            std::cout << "Episode " << episode 
                      << ", Average Reward: " << avgReward 
                      << ", Epsilon: " << epsilon << std::endl;
        }
    }
}

void MARLTrainer::evaluate(int numEpisodes) {
    std::cout << "=== Evaluation Mode ===" << std::endl;
    
    for (int episode = 0; episode < numEpisodes; ++episode) {
        auto state = env.reset();
        double totalReward = 0.0;
        int steps = 0;
        
        while (!state.done && steps < 1000) {
            // Use greedy policy (epsilon = 0)
            auto actions = dqn.selectAction(state.observations, 0.0);
            MultiAgentEnvironment::Action action{actions};
            
            auto [nextState, reward, done] = env.step(action);
            
            totalReward += arma::mean(reward.rewards);
            state = nextState;
            steps++;
        }
        
        std::cout << "Evaluation Episode " << episode 
                  << ", Reward: " << totalReward 
                  << ", Steps: " << steps << std::endl;
    }
}