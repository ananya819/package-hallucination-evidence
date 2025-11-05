#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>
#include <mlpack/methods/reinforcement_learning/replay/random_replay.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
using namespace arma;

/**
 * Deep Q-Network (DQN) Agent implementation
 */
template<typename EnvironmentType>
class DQNAgent
{
public:
    DQNAgent(EnvironmentType& environment,
             const size_t hiddenLayerSize = 128,
             const size_t replayBufferSize = 10000,
             const size_t batchSize = 32,
             const double learningRate = 0.001,
             const double discountFactor = 0.99,
             const double initialEpsilon = 1.0,
             const double finalEpsilon = 0.01,
             const size_t epsilonDecaySteps = 10000) :
        environment(environment),
        batchSize(batchSize),
        learningRate(learningRate),
        discountFactor(discountFactor),
        epsilon(initialEpsilon),
        initialEpsilon(initialEpsilon),
        finalEpsilon(finalEpsilon),
        epsilonDecaySteps(epsilonDecaySteps),
        totalSteps(0)
    {
        // Initialize Q-network
        InitializeQNetwork(hiddenLayerSize);
        
        // Initialize target network (copy of Q-network)
        InitializeTargetNetwork();
        
        // Initialize experience replay buffer
        replayBuffer = std::make_unique<RandomReplay<EnvironmentType>>(
            replayBufferSize, batchSize);
    }

    /**
     * Initialize the Q-network architecture
     */
    void InitializeQNetwork(const size_t hiddenLayerSize)
    {
        qNetwork = std::make_unique<FFN<MeanSquaredError<>, HeInitialization>>();
        
        // Input layer size depends on state dimension
        const size_t stateSize = environment.State().size();
        const size_t actionSize = environment.Action().size();
        
        // Add hidden layers
        qNetwork->Add<Linear<>>(stateSize, hiddenLayerSize);
        qNetwork->Add<ReLULayer<>>();
        qNetwork->Add<Linear<>>(hiddenLayerSize, hiddenLayerSize);
        qNetwork->Add<ReLULayer<>>();
        qNetwork->Add<Linear<>>(hiddenLayerSize, actionSize);
        
        // Create Adam optimizer
        optimizer = std::make_unique<ens::Adam>(
            learningRate,  // learning rate
            batchSize,     // batch size
            0.9,          // beta1
            0.999,        // beta2
            1e-8          // epsilon
        );
    }

    /**
     * Initialize target network as a copy of Q-network
     */
    void InitializeTargetNetwork()
    {
        targetNetwork = std::make_unique<FFN<MeanSquaredError<>, HeInitialization>>();
        
        // Copy architecture from Q-network
        const size_t stateSize = environment.State().size();
        const size_t actionSize = environment.Action().size();
        
        targetNetwork->Add<Linear<>>(stateSize, 128);
        targetNetwork->Add<ReLULayer<>>();
        targetNetwork->Add<Linear<>>(128, 128);
        targetNetwork->Add<ReLULayer<>>();
        targetNetwork->Add<Linear<>>(128, actionSize);
        
        // Copy weights from Q-network
        UpdateTargetNetwork();
    }

    /**
     * Select action using epsilon-greedy policy
     */
    typename EnvironmentType::Action SelectAction(const arma::colvec& state)
    {
        // Random number for epsilon-greedy
        double randomValue = arma::randu();
        
        if (randomValue < epsilon)
        {
            // Exploration: random action
            return environment.Action().Sample();
        }
        else
        {
            // Exploitation: best action from Q-network
            return GetBestAction(state);
        }
    }

    /**
     * Get the best action from Q-network
     */
    typename EnvironmentType::Action GetBestAction(const arma::colvec& state)
    {
        arma::colvec qValues;
        qNetwork->Predict(state, qValues);
        
        // Find action with maximum Q-value
        arma::uword bestAction;
        qValues.max(bestAction);
        
        return typename EnvironmentType::Action(bestAction);
    }

    /**
     * Store experience in replay buffer
     */
    void StoreExperience(const arma::colvec& state,
                        const typename EnvironmentType::Action& action,
                        const double reward,
                        const arma::colvec& nextState,
                        const bool done)
    {
        replayBuffer->Store(state, action, reward, nextState, done);
    }

    /**
     * Train the Q-network using experience replay
     */
    double Train()
    {
        if (replayBuffer->Size() < batchSize)
        {
            return 0.0; // Not enough experiences yet
        }
        
        // Sample batch from replay buffer
        arma::mat states, nextStates;
        std::vector<typename EnvironmentType::Action> actions;
        arma::colvec rewards;
        arma::irowvec isTerminal;
        
        replayBuffer->Sample(states, actions, rewards, nextStates, isTerminal);
        
        // Compute target Q-values
        arma::mat targetQValues = ComputeTargetQValues(nextStates, rewards, isTerminal);
        
        // Compute current Q-values for the taken actions
        arma::mat currentQValues;
        qNetwork->Predict(states, currentQValues);
        
        // Update Q-values for the taken actions
        for (size_t i = 0; i < batchSize; ++i)
        {
            size_t actionIndex = static_cast<size_t>(actions[i]);
            currentQValues(actionIndex, i) = targetQValues(i);
        }
        
        // Train the Q-network
        double loss = qNetwork->Evaluate(states, currentQValues);
        qNetwork->Gradient(states, currentQValues);
        optimizer->Update(qNetwork->Parameters().parameters(),
                         qNetwork->Parameters().gradient());
        
        // Update epsilon
        UpdateEpsilon();
        
        // Periodically update target network
        if (totalSteps % targetUpdateFrequency == 0)
        {
            UpdateTargetNetwork();
        }
        
        totalSteps++;
        return loss;
    }

    /**
     * Compute target Q-values using target network
     */
    arma::mat ComputeTargetQValues(const arma::mat& nextStates,
                                  const arma::colvec& rewards,
                                  const arma::irowvec& isTerminal)
    {
        arma::mat targetQValues(batchSize, 1);
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            if (isTerminal(i))
            {
                // Terminal state: target is just the reward
                targetQValues(i) = rewards(i);
            }
            else
            {
                // Non-terminal state: target is reward + discount * max_a Q_target(s', a)
                arma::colvec nextQValues;
                targetNetwork->Predict(nextStates.col(i), nextQValues);
                
                double maxNextQ = arma::max(nextQValues);
                targetQValues(i) = rewards(i) + discountFactor * maxNextQ;
            }
        }
        
        return targetQValues;
    }

    /**
     * Update epsilon for epsilon-greedy policy
     */
    void UpdateEpsilon()
    {
        if (epsilon > finalEpsilon)
        {
            double decay = (initialEpsilon - finalEpsilon) / epsilonDecaySteps;
            epsilon = std::max(finalEpsilon, epsilon - decay);
        }
    }

    /**
     * Update target network weights from Q-network
     */
    void UpdateTargetNetwork()
    {
        targetNetwork->Parameters() = qNetwork->Parameters();
    }

    /**
     * Save model parameters
     */
    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "q_network", qNetwork->Parameters().parameters());
    }

    /**
     * Load model parameters
     */
    void LoadModel(const std::string& filename)
    {
        arma::mat parameters;
        data::Load(filename, "q_network", parameters);
        qNetwork->Parameters().parameters() = parameters;
        UpdateTargetNetwork();
    }

    // Getters
    double GetEpsilon() const { return epsilon; }
    size_t GetTotalSteps() const { return totalSteps; }
    size_t GetReplayBufferSize() const { return replayBuffer->Size(); }

private:
    EnvironmentType& environment;
    size_t batchSize;
    double learningRate;
    double discountFactor;
    double epsilon;
    double initialEpsilon;
    double finalEpsilon;
    size_t epsilonDecaySteps;
    size_t totalSteps;
    const size_t targetUpdateFrequency = 1000; // Update target network every 1000 steps
    
    std::unique_ptr<FFN<MeanSquaredError<>, HeInitialization>> qNetwork;
    std::unique_ptr<FFN<MeanSquaredError<>, HeInitialization>> targetNetwork;
    std::unique_ptr<ens::Adam> optimizer;
    std::unique_ptr<RandomReplay<EnvironmentType>> replayBuffer;
};

/**
 * Example Environment: CartPole (simplified)
 */
class CartPoleEnvironment
{
public:
    class Action
    {
    public:
        enum : size_t
        {
            LEFT = 0,
            RIGHT = 1
        };
        
        Action(size_t action) : action(action) {}
        operator size_t() const { return action; }
        
        static Action Sample()
        {
            return Action(arma::randi() % 2);
        }
        
        size_t size() const { return 2; }
        
    private:
        size_t action;
    };
    
    CartPoleEnvironment()
    {
        // Initialize state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
        state = {0.0, 0.0, 0.0, 0.0};
        steps = 0;
        maxSteps = 200;
    }
    
    void Reset()
    {
        state = {0.0, 0.0, 0.0, 0.0};
        steps = 0;
    }
    
    double Step(const Action& action)
    {
        // Simplified cart-pole dynamics
        double force = (action == Action::LEFT) ? -1.0 : 1.0;
        
        // Update state using simplified physics
        state[0] += 0.02 * state[1]; // position
        state[1] += 0.02 * (force - 0.1 * state[1]); // velocity
        state[2] += 0.02 * state[3]; // angle
        state[3] += 0.02 * (9.8 * std::sin(state[2]) - 0.1 * state[3] + force * std::cos(state[2])); // angular velocity
        
        steps++;
        
        // Check termination conditions
        bool done = (std::abs(state[0]) > 2.4 || 
                    std::abs(state[2]) > (12 * 3.14159 / 180) ||
                    steps >= maxSteps);
        
        // Reward: +1 for each step
        double reward = done ? 0.0 : 1.0;
        
        if (done)
        {
            Reset();
        }
        
        return reward;
    }
    
    const arma::colvec& State() const
    {
        return state;
    }
    
    Action Action() const
    {
        return Action::LEFT; // Return default action for size query
    }
    
    size_t StateSize() const { return 4; }
    size_t ActionSize() const { return 2; }

private:
    arma::colvec state;
    size_t steps;
    size_t maxSteps;
};

/**
 * Training loop for DQN agent
 */
void TrainDQNAgent()
{
    // Create environment
    CartPoleEnvironment environment;
    
    // Create DQN agent
    DQNAgent<CartPoleEnvironment> agent(
        environment,           // environment
        128,                   // hidden layer size
        10000,                 // replay buffer size
        32,                    // batch size
        0.001,                 // learning rate
        0.99,                  // discount factor
        1.0,                   // initial epsilon
        0.01,                  // final epsilon
        10000                  // epsilon decay steps
    );
    
    // Training parameters
    const size_t totalEpisodes = 1000;
    const size_t maxStepsPerEpisode = 200;
    
    std::vector<double> episodeRewards;
    
    std::cout << "Training DQN Agent..." << std::endl;
    
    for (size_t episode = 0; episode < totalEpisodes; ++episode)
    {
        environment.Reset();
        double episodeReward = 0.0;
        
        for (size_t step = 0; step < maxStepsPerEpisode; ++step)
        {
            // Get current state
            arma::colvec state = environment.State();
            
            // Select action
            CartPoleEnvironment::Action action = agent.SelectAction(state);
            
            // Take action
            double reward = environment.Step(action);
            episodeReward += reward;
            
            // Get next state
            arma::colvec nextState = environment.State();
            
            // Store experience
            agent.StoreExperience(state, action, reward, nextState, false);
            
            // Train agent
            double loss = agent.Train();
            
            if (step == maxStepsPerEpisode - 1)
            {
                break;
            }
        }
        
        episodeRewards.push_back(episodeReward);
        
        // Log progress
        if (episode % 100 == 0)
        {
            double avgReward = std::accumulate(
                episodeRewards.end() - std::min(size_t(100), episodeRewards.size()),
                episodeRewards.end(), 0.0) / std::min(size_t(100), episodeRewards.size());
            
            std::cout << "Episode: " << episode 
                      << ", Avg Reward: " << avgReward
                      << ", Epsilon: " << agent.GetEpsilon()
                      << ", Replay Size: " << agent.GetReplayBufferSize()
                      << std::endl;
        }
    }
    
    // Save trained model
    agent.SaveModel("dqn_model.xml");
    std::cout << "Training completed! Model saved." << std::endl;
}

/**
 * Example of using the trained agent
 */
void RunTrainedAgent()
{
    CartPoleEnvironment environment;
    DQNAgent<CartPoleEnvironment> agent(environment);
    
    // Load trained model
    agent.LoadModel("dqn_model.xml");
    
    // Run episodes with the trained agent
    for (size_t episode = 0; episode < 5; ++episode)
    {
        environment.Reset();
        double totalReward = 0.0;
        
        for (size_t step = 0; step < 200; ++step)
        {
            arma::colvec state = environment.State();
            CartPoleEnvironment::Action action = agent.GetBestAction(state);
            double reward = environment.Step(action);
            totalReward += reward;
            
            std::cout << "Step: " << step 
                      << ", Action: " << (action == CartPoleEnvironment::Action::LEFT ? "LEFT" : "RIGHT")
                      << ", Reward: " << reward
                      << ", Total Reward: " << totalReward << std::endl;
        }
        
        std::cout << "Episode " << episode << " completed with total reward: " << totalReward << std::endl;
    }
}

int main()
{
    std::cout << "Deep Reinforcement Learning with DQN" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Train the DQN agent
    TrainDQNAgent();
    
    // Run the trained agent
    std::cout << "\nRunning trained agent..." << std::endl;
    RunTrainedAgent();
    
    return 0;
}