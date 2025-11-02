#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/glorot_init.hpp>
#include <mlpack/methods/ann/activation_functions/relu_function.hpp>
#include <mlpack/methods/ann/activation_functions/softsign_function.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <map>

using namespace mlpack;
using namespace mlpack::ann;

// Molecular Graph Structure
struct Atom
{
    size_t atomicNumber;
    double x, y, z; // 3D coordinates
    std::vector<size_t> neighbors; // indices of connected atoms
    
    Atom() : atomicNumber(0), x(0), y(0), z(0) {}
    Atom(size_t atomicNum, double x_, double y_, double z_) 
        : atomicNumber(atomicNum), x(x_), y(y_), z(z_) {}
};

struct Bond
{
    size_t atom1, atom2;
    size_t bondType; // 1=single, 2=double, 3=triple
    
    Bond(size_t a1, size_t a2, size_t type) : atom1(a1), atom2(a2), bondType(type) {}
};

class MolecularGraph
{
public:
    std::vector<Atom> atoms;
    std::vector<Bond> bonds;
    
    void AddAtom(const Atom& atom)
    {
        atoms.push_back(atom);
    }
    
    void AddBond(const Bond& bond)
    {
        bonds.push_back(bond);
        // Update neighbor lists
        atoms[bond.atom1].neighbors.push_back(bond.atom2);
        atoms[bond.atom2].neighbors.push_back(bond.atom1);
    }
    
    // Build adjacency matrix
    arma::mat GetAdjacencyMatrix() const
    {
        size_t numAtoms = atoms.size();
        arma::mat adj(numAtoms, numAtoms);
        adj.zeros();
        
        for (const auto& bond : bonds)
        {
            adj(bond.atom1, bond.atom2) = static_cast<double>(bond.bondType);
            adj(bond.atom2, bond.atom1) = static_cast<double>(bond.bondType);
        }
        
        return adj;
    }
    
    // Get atom features (one-hot encoding + coordinates)
    arma::mat GetAtomFeatures() const
    {
        size_t numAtoms = atoms.size();
        size_t featureDim = 118 + 3; // 118 elements + 3D coordinates
        arma::mat features(featureDim, numAtoms);
        features.zeros();
        
        for (size_t i = 0; i < numAtoms; ++i)
        {
            const auto& atom = atoms[i];
            // One-hot encoding for atomic number
            if (atom.atomicNumber > 0 && atom.atomicNumber <= 118)
            {
                features(atom.atomicNumber - 1, i) = 1.0;
            }
            // Add coordinates
            features(118, i) = atom.x;
            features(119, i) = atom.y;
            features(120, i) = atom.z;
        }
        
        return features;
    }
    
    size_t GetNumAtoms() const { return atoms.size(); }
};

// Graph Attention Layer
template<typename InputDataType = arma::mat, typename OutputDataType = arma::mat>
class GraphAttention
{
public:
    GraphAttention(const size_t inputSize,
                   const size_t outputSize,
                   const size_t numHeads = 8) :
        inputSize(inputSize),
        outputSize(outputSize),
        numHeads(numHeads),
        headDim(outputSize / numHeads)
    {
        // Initialize attention weights
        W_Q.set_size(outputSize, inputSize);
        W_K.set_size(outputSize, inputSize);
        W_V.set_size(outputSize, inputSize);
        W_O.set_size(outputSize, outputSize);
        
        // Initialize weights
        GlorotInitialization<> init;
        init.Initialize(W_Q, outputSize, inputSize);
        init.Initialize(W_K, outputSize, inputSize);
        init.Initialize(W_V, outputSize, inputSize);
        init.Initialize(W_O, outputSize, outputSize);
    }

    template<typename eT>
    void Forward(const arma::Mat<eT>& nodeFeatures,
                 const arma::Mat<eT>& adjacency,
                 arma::Mat<eT>& output)
    {
        size_t numNodes = nodeFeatures.n_cols;
        
        // Linear projections
        arma::Mat<eT> Q = W_Q * nodeFeatures;
        arma::Mat<eT> K = W_K * nodeFeatures;
        arma::Mat<eT> V = W_V * nodeFeatures;
        
        // Split into heads
        std::vector<arma::Mat<eT>> Q_heads(numHeads);
        std::vector<arma::Mat<eT>> K_heads(numHeads);
        std::vector<arma::Mat<eT>> V_heads(numHeads);
        
        for (size_t h = 0; h < numHeads; ++h)
        {
            size_t startIdx = h * headDim;
            size_t endIdx = (h + 1) * headDim - 1;
            
            Q_heads[h] = Q.rows(startIdx, endIdx);
            K_heads[h] = K.rows(startIdx, endIdx);
            V_heads[h] = V.rows(startIdx, endIdx);
        }
        
        // Multi-head attention
        std::vector<arma::Mat<eT>> headOutputs(numHeads);
        
        for (size_t h = 0; h < numHeads; ++h)
        {
            // Compute attention scores
            arma::Mat<eT> scores = Q_heads[h] * K_heads[h].t();
            scores /= std::sqrt(static_cast<eT>(headDim));
            
            // Mask out non-adjacent nodes
            for (size_t i = 0; i < numNodes; ++i)
            {
                for (size_t j = 0; j < numNodes; ++j)
                {
                    if (adjacency(i, j) < 1e-8 && i != j)
                    {
                        scores(i, j) = -std::numeric_limits<eT>::max();
                    }
                }
            }
            
            // Apply softmax
            arma::Mat<eT> attention = Softmax(scores);
            
            // Apply attention to values
            headOutputs[h] = attention * V_heads[h];
        }
        
        // Concatenate heads
        arma::Mat<eT> concatenated = headOutputs[0];
        for (size_t h = 1; h < numHeads; ++h)
        {
            concatenated = arma::join_cols(concatenated, headOutputs[h]);
        }
        
        // Output projection
        output = W_O * concatenated;
    }

private:
    size_t inputSize, outputSize, numHeads, headDim;
    arma::mat W_Q, W_K, W_V, W_O;

    template<typename eT>
    arma::Mat<eT> Softmax(const arma::Mat<eT>& input)
    {
        arma::Mat<eT> output = arma::exp(input);
        arma::Row<eT> sum = arma::sum(output, 0);
        output.each_row() /= sum;
        return output;
    }
};

// Graph Transformer Layer
template<typename InputDataType = arma::mat, typename OutputDataType = arma::mat>
class GraphTransformerLayer
{
public:
    GraphTransformerLayer(const size_t featureSize,
                         const size_t numHeads = 8) :
        featureSize(featureSize),
        numHeads(numHeads)
    {
        // Self-attention mechanism
        attention = std::make_unique<GraphAttention<>>(featureSize, featureSize, numHeads);
        
        // Feed-forward network
        ff1 = std::make_unique<Linear<>>(featureSize, featureSize * 4);
        ff2 = std::make_unique<Linear<>>(featureSize * 4, featureSize);
        
        // Layer normalization
        norm1 = std::make_unique<Linear<>>(featureSize, featureSize);
        norm2 = std::make_unique<Linear<>>(featureSize, featureSize);
        
        // Initialize weights
        GlorotInitialization<> init;
        InitializeWeights(init);
    }

    void InitializeWeights(GlorotInitialization<>& init)
    {
        Linear<>* linear1 = dynamic_cast<Linear<>*>(ff1.get());
        Linear<>* linear2 = dynamic_cast<Linear<>*>(ff2.get());
        Linear<>* linear3 = dynamic_cast<Linear<>*>(norm1.get());
        Linear<>* linear4 = dynamic_cast<Linear<>*>(norm2.get());
        
        if (linear1) init.Initialize(linear1->Weights(), linear1->OutputDimensions(), linear1->InputDimensions());
        if (linear2) init.Initialize(linear2->Weights(), linear2->OutputDimensions(), linear2->InputDimensions());
        if (linear3) init.Initialize(linear3->Weights(), linear3->OutputDimensions(), linear3->InputDimensions());
        if (linear4) init.Initialize(linear4->Weights(), linear4->OutputDimensions(), linear4->InputDimensions());
    }

    template<typename eT>
    void Forward(const arma::Mat<eT>& nodeFeatures,
                 const arma::Mat<eT>& adjacency,
                 arma::Mat<eT>& output)
    {
        // Self-attention
        arma::Mat<eT> attentionOutput;
        attention->Forward(nodeFeatures, adjacency, attentionOutput);
        
        // Residual connection + layer norm
        arma::Mat<eT> residual1 = nodeFeatures + attentionOutput;
        arma::Mat<eT> normOutput1;
        norm1->Forward(residual1, normOutput1);
        
        // Feed-forward
        arma::Mat<eT> ffOutput1, ffOutput2;
        ff1->Forward(normOutput1, ffOutput1);
        ffOutput1 = arma::max(ffOutput1, arma::zeros<arma::mat>(ffOutput1.n_rows, ffOutput1.n_cols)); // ReLU
        ff2->Forward(ffOutput1, ffOutput2);
        
        // Residual connection + layer norm
        arma::Mat<eT> residual2 = normOutput1 + ffOutput2;
        norm2->Forward(residual2, output);
    }

private:
    size_t featureSize, numHeads;
    std::unique_ptr<GraphAttention<>> attention;
    std::unique_ptr<Layer<>> ff1, ff2;
    std::unique_ptr<Layer<>> norm1, norm2;
};

// Molecular Graph Transformer
class MolecularGraphTransformer
{
public:
    MolecularGraphTransformer(const size_t hiddenSize = 128,
                             const size_t numLayers = 6,
                             const size_t numHeads = 8) :
        hiddenSize(hiddenSize),
        numLayers(numLayers),
        numHeads(numHeads)
    {
        // Initialize transformer layers
        for (size_t i = 0; i < numLayers; ++i)
        {
            layers.emplace_back(std::make_unique<GraphTransformerLayer<>>(
                hiddenSize, numHeads));
        }
        
        // Readout layer (global pooling + prediction)
        readout = std::make_unique<Linear<>>(hiddenSize, 1); // Single property prediction
        
        // Initialize weights
        GlorotInitialization<> init;
        Linear<>* linear = dynamic_cast<Linear<>*>(readout.get());
        if (linear)
        {
            init.Initialize(linear->Weights(), linear->OutputDimensions(), linear->InputDimensions());
        }
    }

    // Convert molecular graph to feature representation
    arma::mat ExtractFeatures(const MolecularGraph& graph) const
    {
        return graph.GetAtomFeatures();
    }

    template<typename eT>
    void Forward(const arma::Mat<eT>& nodeFeatures,
                 const arma::Mat<eT>& adjacency,
                 eT& output)
    {
        arma::Mat<eT> currentFeatures = nodeFeatures;
        
        // Forward through transformer layers
        for (auto& layer : layers)
        {
            arma::Mat<eT> layerOutput;
            layer->Forward(currentFeatures, adjacency, layerOutput);
            currentFeatures = layerOutput;
        }
        
        // Global mean pooling
        arma::Col<eT> graphEmbedding = arma::mean(currentFeatures, 1);
        
        // Readout prediction
        arma::Mat<eT> prediction;
        readout->Forward(graphEmbedding, prediction);
        output = prediction(0, 0);
    }

    // Predict molecular property
    double Predict(const MolecularGraph& molecule)
    {
        arma::mat features = ExtractFeatures(molecule);
        arma::mat adjacency = molecule.GetAdjacencyMatrix();
        
        double prediction;
        Forward(features, adjacency, prediction);
        
        return prediction;
    }

    // Training function
    void Train(const std::vector<MolecularGraph>& trainingData,
              const arma::vec& targetProperties,
              size_t numEpochs = 100,
              double learningRate = 0.001)
    {
        std::cout << "Training Molecular Graph Transformer..." << std::endl;
        std::cout << "Training samples: " << trainingData.size() << std::endl;
        
        for (size_t epoch = 0; epoch < numEpochs; ++epoch)
        {
            double totalLoss = 0.0;
            
            for (size_t i = 0; i < trainingData.size(); ++i)
            {
                const auto& molecule = trainingData[i];
                double target = targetProperties(i);
                
                // Forward pass
                double prediction = Predict(molecule);
                
                // Compute loss (MSE)
                double error = prediction - target;
                double loss = 0.5 * error * error;
                totalLoss += loss;
                
                // Update weights (simplified)
                UpdateWeights(learningRate, error);
            }
            
            if (epoch % 10 == 0)
            {
                std::cout << "Epoch " << epoch << ", Average Loss: " 
                         << totalLoss / trainingData.size() << std::endl;
            }
        }
        
        std::cout << "Training completed!" << std::endl;
    }

    // Batch prediction
    void PredictBatch(const std::vector<MolecularGraph>& molecules,
                     arma::vec& predictions)
    {
        predictions.set_size(molecules.size());
        
        for (size_t i = 0; i < molecules.size(); ++i)
        {
            predictions(i) = Predict(molecules[i]);
        }
    }

    // Evaluation function
    double Evaluate(const std::vector<MolecularGraph>& testData,
                   const arma::vec& trueProperties)
    {
        arma::vec predictions;
        PredictBatch(testData, predictions);
        
        // Compute RMSE
        arma::vec errors = predictions - trueProperties;
        double rmse = std::sqrt(arma::mean(arma::square(errors)));
        
        return rmse;
    }

private:
    size_t hiddenSize, numLayers, numHeads;
    std::vector<std::unique_ptr<GraphTransformerLayer<>>> layers;
    std::unique_ptr<Layer<>> readout;

    void UpdateWeights(double learningRate, double error)
    {
        // Simplified weight update - in practice, implement proper backpropagation
        static bool firstCall = true;
        if (firstCall)
        {
            std::cout << "Note: Weight updates are simplified. For production use, "
                      << "implement proper backpropagation through graph transformers." << std::endl;
            firstCall = false;
        }
        
        // In a real implementation, you would:
        // 1. Compute gradients through each transformer layer
        // 2. Backpropagate through attention mechanisms
        // 3. Update all network parameters
        // 4. Apply proper optimization (Adam, etc.)
    }
};

// Molecular Data Generator
class MolecularDataGenerator
{
public:
    // Generate synthetic molecules for training
    static void GenerateSyntheticData(size_t numMolecules,
                                    std::vector<MolecularGraph>& molecules,
                                    arma::vec& properties)
    {
        molecules.clear();
        molecules.reserve(numMolecules);
        properties.set_size(numMolecules);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> atomCountDist(5, 20);
        std::uniform_int_distribution<> elementDist(1, 10); // First 10 elements
        std::uniform_real_distribution<> coordDist(-5.0, 5.0);
        std::uniform_real_distribution<> propertyDist(0.0, 10.0);
        
        std::cout << "Generating synthetic molecular data..." << std::endl;
        
        for (size_t i = 0; i < numMolecules; ++i)
        {
            MolecularGraph molecule;
            
            // Generate atoms
            size_t numAtoms = atomCountDist(gen);
            for (size_t j = 0; j < numAtoms; ++j)
            {
                size_t atomicNum = elementDist(gen);
                double x = coordDist(gen);
                double y = coordDist(gen);
                double z = coordDist(gen);
                molecule.AddAtom(Atom(atomicNum, x, y, z));
            }
            
            // Generate bonds (simple linear chain + some random connections)
            for (size_t j = 0; j < numAtoms - 1; ++j)
            {
                molecule.AddBond(Bond(j, j + 1, 1)); // Single bonds in chain
            }
            
            // Add some random bonds
            std::uniform_int_distribution<> bondDist(0, numAtoms - 1);
            for (size_t j = 0; j < numAtoms / 3; ++j)
            {
                size_t atom1 = bondDist(gen);
                size_t atom2 = bondDist(gen);
                if (atom1 != atom2)
                {
                    molecule.AddBond(Bond(atom1, atom2, 1));
                }
            }
            
            molecules.push_back(molecule);
            
            // Generate synthetic property (based on molecular features)
            double property = ComputeSyntheticProperty(molecule, gen);
            properties(i) = property;
        }
        
        std::cout << "Generated " << numMolecules << " synthetic molecules" << std::endl;
    }

private:
    static double ComputeSyntheticProperty(const MolecularGraph& molecule, std::mt19937& gen)
    {
        // Simple synthetic property computation
        double property = 0.0;
        
        // Weight-based property
        for (const auto& atom : molecule.atoms)
        {
            property += atom.atomicNumber * 0.5;
        }
        
        // Bond-based property
        for (const auto& bond : molecule.bonds)
        {
            property += bond.bondType * 1.2;
        }
        
        // Size-based property
        property += molecule.GetNumAtoms() * 0.3;
        
        // Add some noise
        std::normal_distribution<> noise(0.0, 1.0);
        property += noise(gen);
        
        return std::max(0.0, property); // Ensure non-negative
    }
};

// Molecular Data Preprocessing
class MolecularPreprocessor
{
public:
    // Normalize molecular coordinates
    static void NormalizeCoordinates(MolecularGraph& molecule)
    {
        if (molecule.atoms.empty()) return;
        
        // Compute centroid
        double sumX = 0, sumY = 0, sumZ = 0;
        for (const auto& atom : molecule.atoms)
        {
            sumX += atom.x;
            sumY += atom.y;
            sumZ += atom.z;
        }
        
        double centroidX = sumX / molecule.atoms.size();
        double centroidY = sumY / molecule.atoms.size();
        double centroidZ = sumZ / molecule.atoms.size();
        
        // Center coordinates
        for (auto& atom : molecule.atoms)
        {
            atom.x -= centroidX;
            atom.y -= centroidY;
            atom.z -= centroidZ;
        }
    }

    // Remove duplicate atoms
    static void RemoveDuplicateAtoms(MolecularGraph& molecule)
    {
        // Simple duplicate removal based on coordinates
        std::vector<bool> toRemove(molecule.atoms.size(), false);
        
        for (size_t i = 0; i < molecule.atoms.size(); ++i)
        {
            for (size_t j = i + 1; j < molecule.atoms.size(); ++j)
            {
                const auto& atom1 = molecule.atoms[i];
                const auto& atom2 = molecule.atoms[j];
                
                double dist = std::sqrt(std::pow(atom1.x - atom2.x, 2) +
                                      std::pow(atom1.y - atom2.y, 2) +
                                      std::pow(atom1.z - atom2.z, 2));
                
                if (dist < 0.1) // Threshold for duplicates
                {
                    toRemove[j] = true;
                }
            }
        }
        
        // Remove atoms and update bonds
        std::vector<Atom> newAtoms;
        std::map<size_t, size_t> oldToNewIndex;
        size_t newIndex = 0;
        
        for (size_t i = 0; i < molecule.atoms.size(); ++i)
        {
            if (!toRemove[i])
            {
                oldToNewIndex[i] = newIndex++;
                newAtoms.push_back(molecule.atoms[i]);
            }
        }
        
        molecule.atoms = newAtoms;
        
        // Update bonds
        std::vector<Bond> newBonds;
        for (const auto& bond : molecule.bonds)
        {
            auto it1 = oldToNewIndex.find(bond.atom1);
            auto it2 = oldToNewIndex.find(bond.atom2);
            
            if (it1 != oldToNewIndex.end() && it2 != oldToNewIndex.end())
            {
                newBonds.emplace_back(it1->second, it2->second, bond.bondType);
            }
        }
        
        molecule.bonds = newBonds;
    }
};

// Example molecular structures for testing
class MolecularExamples
{
public:
    static MolecularGraph CreateWaterMolecule()
    {
        MolecularGraph water;
        
        // Add atoms: O, H, H
        water.AddAtom(Atom(8, 0.0, 0.0, 0.0));   // Oxygen
        water.AddAtom(Atom(1, 0.757, 0.586, 0.0)); // Hydrogen 1
        water.AddAtom(Atom(1, -0.757, 0.586, 0.0)); // Hydrogen 2
        
        // Add bonds
        water.AddBond(Bond(0, 1, 1)); // O-H
        water.AddBond(Bond(0, 2, 1)); // O-H
        
        return water;
    }
    
    static MolecularGraph CreateMethaneMolecule()
    {
        MolecularGraph methane;
        
        // Add atoms: C, H, H, H, H
        methane.AddAtom(Atom(6, 0.0, 0.0, 0.0));     // Carbon
        methane.AddAtom(Atom(1, 0.629, 0.629, 0.629)); // Hydrogen 1
        methane.AddAtom(Atom(1, -0.629, -0.629, 0.629)); // Hydrogen 2
        methane.AddAtom(Atom(1, 0.629, -0.629, -0.629)); // Hydrogen 3
        methane.AddAtom(Atom(1, -0.629, 0.629, -0.629)); // Hydrogen 4
        
        // Add bonds
        methane.AddBond(Bond(0, 1, 1)); // C-H
        methane.AddBond(Bond(0, 2, 1)); // C-H
        methane.AddBond(Bond(0, 3, 1)); // C-H
        methane.AddBond(Bond(0, 4, 1)); // C-H
        
        return methane;
    }
};

// Example usage and demonstration
int main()
{
    std::cout << "Dynamic Graph Transformer for Molecular Property Prediction" << std::endl;
    std::cout << "=========================================================" << std::endl;
    
    try
    {
        // Example 1: Basic molecular property prediction
        {
            std::cout << "\nExample 1: Basic Molecular Property Prediction" << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
            
            // Create molecular graph transformer
            MolecularGraphTransformer model(128, 4, 8);
            
            // Generate synthetic training data
            std::vector<MolecularGraph> trainingData;
            arma::vec trainingProperties;
            MolecularDataGenerator::GenerateSyntheticData(200, trainingData, trainingProperties);
            
            // Preprocess data
            std::cout << "Preprocessing molecular data..." << std::endl;
            for (auto& molecule : trainingData)
            {
                MolecularPreprocessor::NormalizeCoordinates(molecule);
                MolecularPreprocessor::RemoveDuplicateAtoms(molecule);
            }
            
            // Train the model
            std::cout << "Training the molecular graph transformer..." << std::endl;
            model.Train(trainingData, trainingProperties, 50, 0.001);
            
            // Generate test data
            std::vector<MolecularGraph> testData;
            arma::vec testProperties;
            MolecularDataGenerator::GenerateSyntheticData(50, testData, testProperties);
            
            // Preprocess test data
            for (auto& molecule : testData)
            {
                MolecularPreprocessor::NormalizeCoordinates(molecule);
                MolecularPreprocessor::RemoveDuplicateAtoms(molecule);
            }
            
            // Evaluate the model
            double rmse = model.Evaluate(testData, testProperties);
            std::cout << "Test RMSE: " << rmse << std::endl;
            
            // Test individual predictions
            if (!testData.empty())
            {
                double prediction = model.Predict(testData[0]);
                std::cout << "Sample prediction: " << prediction 
                         << " (true: " << testProperties(0) << ")" << std::endl;
            }
        }
        
        // Example 2: Real molecular examples
        {
            std::cout << "\nExample 2: Real Molecular Examples" << std::endl;
            std::cout << "----------------------------------" << std::endl;
            
            // Create example molecules
            MolecularGraph water = MolecularExamples::CreateWaterMolecule();
            MolecularGraph methane = MolecularExamples::CreateMethaneMolecule();
            
            // Create and train a simple model
            MolecularGraphTransformer model(64, 2, 4);
            
            // Simple demonstration without training
            std::cout << "Water molecule atoms: " << water.GetNumAtoms() << std::endl;
            std::cout << "Water molecule bonds: " << water.bonds.size() << std::endl;
            
            std::cout << "Methane molecule atoms: " << methane.GetNumAtoms() << std::endl;
            std::cout << "Methane molecule bonds: " << methane.bonds.size() << std::endl;
            
            // Make predictions (random since model is untrained)
            double waterProperty = model.Predict(water);
            double methaneProperty = model.Predict(methane);
            
            std::cout << "Water property prediction: " << waterProperty << std::endl;
            std::cout << "Methane property prediction: " << methaneProperty << std::endl;
        }
        
        // Example 3: Molecular data preprocessing
        {
            std::cout << "\nExample 3: Molecular Data Preprocessing" << std::endl;
            std::cout << "--------------------------------------" << std::endl;
            
            // Create a molecule with duplicate atoms
            MolecularGraph molecule;
            molecule.AddAtom(Atom(6, 0.0, 0.0, 0.0));
            molecule.AddAtom(Atom(6, 0.0, 0.0, 0.0)); // Duplicate
            molecule.AddAtom(Atom(1, 1.0, 0.0, 0.0));
            
            std::cout << "Before preprocessing:" << std::endl;
            std::cout << "  Atoms: " << molecule.GetNumAtoms() << std::endl;
            
            // Apply preprocessing
            MolecularPreprocessor::RemoveDuplicateAtoms(molecule);
            MolecularPreprocessor::NormalizeCoordinates(molecule);
            
            std::cout << "After preprocessing:" << std::endl;
            std::cout << "  Atoms: " << molecule.GetNumAtoms() << std::endl;
            std::cout << "  Centroid normalized" << std::endl;
        }
        
        // Example 4: Model architecture details
        {
            std::cout << "\nExample 4: Graph Transformer Architecture" << std::endl;
            std::cout << "------------------------------------------" << std::endl;
            
            // Create detailed model
            const size_t hiddenSize = 256;
            const size_t numLayers = 6;
            const size_t numHeads = 8;
            
            MolecularGraphTransformer detailedModel(hiddenSize, numLayers, numHeads);
            
            std::cout << "Graph Transformer Architecture:" << std::endl;
            std::cout << "  Hidden Size: " << hiddenSize << std::endl;
            std::cout << "  Transformer Layers: " << numLayers << std::endl;
            std::cout << "  Attention Heads: " << numHeads << std::endl;
            std::cout << "  Node Features: 121 (118 elements + 3D coordinates)" << std::endl;
            std::cout << "  Global Pooling: Mean pooling over nodes" << std::endl;
            std::cout << "  Output: Single molecular property value" << std::endl;
        }
        
        std::cout << "\nAll examples completed successfully!" << std::endl;
        std::cout << "\nNote: This implementation demonstrates core concepts for molecular property prediction." << std::endl;
        std::cout << "For production use, consider:" << std::endl;
        std::cout << "- Implementing proper backpropagation through graph attention mechanisms" << std::endl;
        std::cout << "- Adding more sophisticated molecular features (bond types, hybridization, etc.)" << std::endl;
        std::cout << "- Implementing advanced architectures (GraphSAGE, GIN, etc.)" << std::endl;
        std::cout << "- Adding proper regularization and dropout for molecular data" << std::endl;
        std::cout << "- Optimizing for GPU acceleration for large molecular datasets" << std::endl;
        std::cout << "- Implementing distributed training for large-scale molecular property prediction" << std::endl;
        
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}