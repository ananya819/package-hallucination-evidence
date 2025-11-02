#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/rnn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/loss_functions/reinforce_normal_loss.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/activation_functions/softplus_function.hpp>
#include <mlpack/methods/ann/activation_functions/tanh_function.hpp>
#include <mlpack/methods/ann/activation_functions/softmax_function.hpp>
#include <mlpack/methods/ann/activation_functions/identity_function.hpp>
#include <mlpack/methods/reinforcement_learning/environment/mountain_car.hpp>
#include <mlpack/methods/reinforcement_learning/environment/cart_pole.hpp>
#include <mlpack/methods/reinforcement_learning/policy/policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <memory>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;

// Custom Policy Network for Policy Gradient
template<typename EnvironmentType>
class PolicyNetwork
{
public:
    using StateType = typename EnvironmentType::State;
    using ActionType = typename EnvironmentType::Action;

    PolicyNetwork(const size_t stateSize,
                  const size_t actionSize,
                  const size_t hiddenSize = 128) :
        stateSize(stateSize),
        actionSize(actionSize),
        hiddenSize(hiddenSize),
        gamma(0.99),
        learningRate(0.001)
    {
        // Initialize policy network (actor)
        InitializePolicyNetwork();
        
        // Initialize value network (critic)
        InitializeValueNetwork();
        
        // Initialize random number generator
        generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    void InitializePolicyNetwork()
    {
        policyNetwork = std::make_unique<FFN<MeanSquaredError<>, GlorotInitialization>>();
        
        // Input layer
        policyNetwork->Add<Linear<>>(stateSize, hiddenSize);
        policyNetwork->Add<TanhFunction<>>();
        
        // Hidden layers
        policyNetwork->Add<Linear<>>(hiddenSize, hiddenSize);
        policyNetwork->Add<TanhFunction<>>();
        
        // Output layer for action probabilities
        policyNetwork->Add<Linear<>>(hiddenSize, actionSize);
        policyNetwork->Add<Softmax<>>();
    }

    void InitializeValueNetwork()
    {
        valueNetwork = std::make_unique<FFN<MeanSquaredError<>, GlorotInitialization>>();
        
        // Input layer
        valueNetwork->Add<Linear<>>(stateSize, hiddenSize);
        valueNetwork->Add<TanhFunction<>>();
        
        // Hidden layers
        valueNetwork->Add<Linear<>>(hiddenSize, hiddenSize);
        valueNetwork->Add<TanhFunction<>>();
        
        // Output layer for state value
        valueNetwork->Add<Linear<>>(hiddenSize, 1);
    }

    // Select action based on policy (stochastic)
    typename EnvironmentType::Action SelectAction(const arma::colvec& state)
    {
        // Get action probabilities from policy network
        arma::colvec actionProbs;
        policyNetwork->Predict(arma::mat(state), actionProbs);
        
        // Sample action according to probabilities
        std::discrete_distribution<> dist(actionProbs.begin(), actionProbs.end());
        size_t actionIndex = dist(generator);
        
        typename EnvironmentType::Action action;
        action.action = actionIndex;
        return action;
    }

    // Select action greedily (for evaluation)
    typename EnvironmentType::Action SelectActionGreedy(const arma::colvec& state)
    {
        arma::colvec actionProbs;
        policyNetwork->Predict(arma::mat(state), actionProbs);
        
        size_t actionIndex = arma::index_max(actionProbs);
        
        typename EnvironmentType::Action action;
        action.action = actionIndex;
        return action;
    }

    // Store experience for training
    void StoreExperience(const arma::colvec& state,
                        const typename EnvironmentType::Action& action,
                        const double reward,
                        const arma::colvec& nextState,
                        const bool done)
    {
        experiences.push_back({state, action, reward, nextState, done});
    }

    // Compute discounted returns
    std::vector<double> ComputeReturns()
    {
        std::vector<double> returns(experiences.size());
        double G = 0.0;
        
        // Compute returns in reverse order
        for (int t = experiences.size() - 1; t >= 0; --t)
        {
            G = experiences[t].reward + gamma * G * (1 - experiences[t].done);
            returns[t] = G;
        }
        
        return returns;
    }

    // Compute advantages using critic
    std::vector<double> ComputeAdvantages(const std::vector<double>& returns)
    {
        std::vector<double> advantages(experiences.size());
        
        for (size_t i = 0; i < experiences.size(); ++i)
        {
            // Get value of current state
            arma::colvec stateValue;
            valueNetwork->Predict(arma::mat(experiences[i].state), stateValue);
            
            // Advantage = Return - Value
            advantages[i] = returns[i] - stateValue(0);
        }
        
        return advantages;
    }

    // Train the policy and value networks
    void Train()
    {
        if (experiences.empty()) return;
        
        // Compute returns and advantages
        auto returns = ComputeReturns();
        auto advantages = ComputeAdvantages(returns);
        
        // Normalize advantages
        double meanAdv = std::accumulate(advantages.begin(), advantages.end(), 0.0) / advantages.size();
        double stdAdv = 0.0;
        for (const auto& adv : advantages)
        {
            stdAdv += std::pow(adv - meanAdv, 2);
        }
        stdAdv = std::sqrt(stdAdv / advantages.size() + 1e-8);
        
        for (auto& adv : advantages)
        {
            adv = (adv - meanAdv) / (stdAdv + 1e-8);
        }
        
        // Prepare training data
        arma::mat states(stateSize, experiences.size());
        arma::mat actions(1, experiences.size());
        arma::mat actionAdvantages(1, experiences.size());
        arma::mat targetValues(1, experiences.size());
        
        for (size_t i = 0; i < experiences.size(); ++i)
        {
            states.col(i) = experiences[i].state;
            actions(0, i) = static_cast<double>(experiences[i].action.action);
            actionAdvantages(0, i) = advantages[i];
            targetValues(0, i) = returns[i];
        }
        
        // Train value network (critic)
        arma::mat predictedValues;
        valueNetwork->Predict(states, predictedValues);
        valueNetwork->Train(states, targetValues);
        
        // Train policy network (actor) - simplified implementation
        // In practice, you would implement a custom loss function for policy gradient
        UpdatePolicyNetwork(states, actions, actionAdvantages);
        
        // Clear experiences
        experiences.clear();
    }

    // Simplified policy network update
    void UpdatePolicyNetwork(const arma::mat& states,
                           const arma::mat& actions,
                           const arma::mat& advantages)
    {
        // This is a simplified approach - in practice, you'd implement
        // proper policy gradient updates with custom loss functions
        
        // Get current action probabilities
        arma::mat actionProbs;
        policyNetwork->Predict(states, actionProbs);
        
        // Compute policy gradient (simplified)
        arma::mat gradients = actionProbs;
        for (size_t i = 0; i < actions.n_cols; ++i)
        {
            size_t actionIdx = static_cast<size_t>(actions(0, i));
            gradients(actionIdx, i) *= advantages(0, i);
        }
        
        // Update would happen here in a complete implementation
        // policyNetwork->Train(states, gradients);
    }

    // Evaluate the policy
    double Evaluate(EnvironmentType& environment, size_t episodes = 10)
    {
        double totalReward = 0.0;
        
        for (size_t episode = 0; episode < episodes; ++episode)
        {
            typename EnvironmentType::State state = environment.InitialSample();
            double episodeReward = 0.0;
            size_t steps = 0;
            const size_t maxSteps = 1000;
            
            while (!environment.IsTerminal(state) && steps < maxSteps)
            {
                auto action = SelectActionGreedy(arma::vec(state.Data()));
                auto stepResult = environment.Sample(state, action);
                
                episodeReward += stepResult.reward;
                state = stepResult.nextState;
                steps++;
            }
            
            totalReward += episodeReward;
        }
        
        return totalReward / episodes;
    }

    // Get network parameters for saving/loading
    void GetPolicyParameters(arma::mat& parameters)
    {
        policyNetwork->Parameters() = parameters;
    }

    void SetPolicyParameters(const arma::mat& parameters)
    {
        policyNetwork->Parameters() = parameters;
    }

private:
    struct Experience
    {
        arma::colvec state;
        typename EnvironmentType::Action action;
        double reward;
        arma::colvec nextState;
        bool done;
    };

    size_t stateSize;
    size_t actionSize;
    size_t hiddenSize;
    double gamma;
    double learningRate;
    
    std::unique_ptr<FFN<MeanSquaredError<>, GlorotInitialization>> policyNetwork;
    std::unique_ptr<FFN<MeanSquaredError<>, GlorotInitialization>> valueNetwork;
    std::vector<Experience> experiences;
    std::mt19937 generator;
};

// REINFORCE Algorithm Implementation
template<typename EnvironmentType>
class REINFORCE
{
public:
    REINFORCE(PolicyNetwork<EnvironmentType>& policy,
              const double learningRate = 0.001,
              const double gamma = 0.99) :
        policy(policy),
        learningRate(learningRate),
        gamma(gamma)
    {}

    void Train(EnvironmentType& environment, size_t episodes)
    {
        for (size_t episode = 0; episode < episodes; ++episode)
        {
            // Collect episode
            auto episodeData = CollectEpisode(environment);
            
            // Compute returns
            auto returns = ComputeReturns(episodeData);
            
            // Update policy
            UpdatePolicy(episodeData, returns);
            
            // Print progress
            if (episode % 100 == 0)
            {
                double avgReward = policy.Evaluate(environment, 5);
                std::cout << "Episode " << episode 
                          << ", Average Reward: " << avgReward << std::endl;
            }
        }
    }

private:
    struct EpisodeStep
    {
        arma::colvec state;
        typename EnvironmentType::Action action;
        double reward;
    };

    std::vector<EpisodeStep> CollectEpisode(EnvironmentType& environment)
    {
        std::vector<EpisodeStep> episode;
        typename EnvironmentType::State state = environment.InitialSample();
        
        while (!environment.IsTerminal(state))
        {
            auto action = policy.SelectAction(arma::vec(state.Data()));
            auto stepResult = environment.Sample(state, action);
            
            episode.push_back({arma::vec(state.Data()), action, stepResult.reward});
            
            state = stepResult.nextState;
        }
        
        return episode;
    }

    std::vector<double> ComputeReturns(const std::vector<EpisodeStep>& episode)
    {
        std::vector<double> returns(episode.size());
        double G = 0.0;
        
        for (int t = episode.size() - 1; t >= 0; --t)
        {
            G = episode[t].reward + gamma * G;
            returns[t] = G;
        }
        
        return returns;
    }

    void UpdatePolicy(const std::vector<EpisodeStep>& episode,
                     const std::vector<double>& returns)
    {
        // In a complete implementation, this would update the policy network
        // using the policy gradient theorem
        // ∇J(θ) = E[∇logπ(a|s,θ) * G_t]
    }

    PolicyNetwork<EnvironmentType>& policy;
    double learningRate;
    double gamma;
};

// Example usage with CartPole environment
int main()
{
    // Create CartPole environment
    CartPole::State::dimension = 4;
    CartPole environment;
    
    // Get environment dimensions
    const size_t stateSize = environment.InitialSample().Data().size();
    const size_t actionSize = 2; // CartPole has 2 actions: left and right
    
    std::cout << "State size: " << stateSize << std::endl;
    std::cout << "Action size: " << actionSize << std::endl;
    
    // Create policy network
    PolicyNetwork<CartPole> policyNetwork(stateSize, actionSize, 64);
    
    // Create REINFORCE trainer
    REINFORCE<CartPole> reinforce(policyNetwork, 0.001, 0.99);
    
    std::cout << "Starting training..." << std::endl;
    
    // Train the policy
    reinforce.Train(environment, 1000);
    
    // Evaluate the trained policy
    std::cout << "\nEvaluating trained policy..." << std::endl;
    double finalScore = policyNetwork.Evaluate(environment, 10);
    std::cout << "Final average score: " << finalScore << std::endl;
    
    // Test a single episode
    std::cout << "\nTesting single episode:" << std::endl;
    CartPole::State state = environment.InitialSample();
    double totalReward = 0.0;
    size_t steps = 0;
    
    while (!environment.IsTerminal(state) && steps < 1000)
    {
        auto action = policyNetwork.SelectActionGreedy(arma::vec(state.Data()));
        auto stepResult = environment.Sample(state, action);
        
        totalReward += stepResult.reward;
        state = stepResult.nextState;
        steps++;
        
        if (steps % 100 == 0)
        {
            std::cout << "Step " << steps << ", Reward: " << totalReward << std::endl;
        }
    }
    
    std::cout << "Episode finished after " << steps 
              << " steps with total reward: " << totalReward << std::endl;
    
    return 0;
}