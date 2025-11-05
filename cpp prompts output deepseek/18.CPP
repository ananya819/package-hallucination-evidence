#include <iostream>
#include <vector>
#include "hierarchical_rl_agent.hpp"

// Simple continuous control environment for demonstration
class SimpleContinuousEnv
{
public:
    SimpleContinuousEnv(size_t stateDim = 4, size_t actionDim = 2) 
        : stateDim(stateDim), actionDim(actionDim)
    {
        Reset();
    }

    void Reset()
    {
        state = arma::zeros(stateDim, 1);
        steps = 0;
    }

    std::tuple<arma::mat, double, bool> Step(const arma::mat& action)
    {
        // Simple dynamics: state = state + action
        arma::mat clippedAction = arma::clamp(action, -1.0, 1.0);
        state += clippedAction * 0.1;
        
        // Simple reward: distance to origin
        double reward = -arma::norm(state, 2);
        
        steps++;
        bool done = steps >= 200;
        
        return {state, reward, done};
    }

    arma::mat GetState() const { return state; }
    size_t StateDim() const { return stateDim; }
    size_t ActionDim() const { return actionDim; }

private:
    size_t stateDim;
    size_t actionDim;
    arma::mat state;
    size_t steps;
};

int main()
{
    // Create environment and agent
    SimpleContinuousEnv env(4, 2);
    HierarchicalRLAgent<SimpleContinuousEnv> agent(
        env.StateDim(),  // state size
        env.ActionDim(), // action size  
        2,              // goal size
        5               // high-level decision interval
    );

    // Training parameters
    const size_t numEpisodes = 1000;
    const size_t maxSteps = 200;

    std::vector<double> episodeRewards;

    // Training loop
    for (size_t episode = 0; episode < numEpisodes; ++episode)
    {
        env.Reset();
        double totalReward = 0.0;

        std::vector<arma::mat> states;
        std::vector<arma::mat> actions;
        std::vector<double> rewards;
        std::vector<arma::mat> nextStates;
        std::vector<bool> dones;

        for (size_t step = 0; step < maxSteps; ++step)
        {
            arma::mat state = env.GetState();
            arma::mat action = agent.SampleAction(state);
            
            auto [nextState, reward, done] = env.Step(action);
            totalReward += reward;

            // Store experience
            states.push_back(state);
            actions.push_back(action);
            rewards.push_back(reward);
            nextStates.push_back(nextState);
            dones.push_back(done);

            if (done) break;
        }

        // Update agent with episode experience
        agent.Update(states, actions, rewards, nextStates, dones);

        episodeRewards.push_back(totalReward);

        // Log progress
        if (episode % 100 == 0)
        {
            double avgReward = std::accumulate(
                episodeRewards.end() - std::min(size_t(100), episodeRewards.size()),
                episodeRewards.end(), 0.0) / 
                std::min(size_t(100), episodeRewards.size());
            
            std::cout << "Episode " << episode 
                      << ", Average Reward: " << avgReward 
                      << ", Current Goal: " << agent.GetGoal().t()
                      << std::endl;
        }
    }

    // Save trained agent
    agent.Save("trained_hierarchical_agent");

    std::cout << "Training completed!" << std::endl;
    return 0;
}