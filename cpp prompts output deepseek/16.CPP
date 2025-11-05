#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include <mlpack/methods/ann/attention/attention.hpp>
#include <armadillo>
#include <vector>
#include <queue>
#include <unordered_map>
#include <memory>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * Graph Data Structure
 * Represents graph with nodes, features, and adjacency information
 */
class GraphData
{
public:
    GraphData(const mat& nodeFeatures, 
              const umat& edges,
              const uvec& labels = uvec()) :
        nodeFeatures(nodeFeatures),
        edges(edges),
        labels(labels),
        numNodes(nodeFeatures.n_cols),
        featureDim(nodeFeatures.n_rows)
    {
        BuildAdjacencyList();
    }

    void BuildAdjacencyList()
    {
        adjacencyList.resize(numNodes);
        
        for (size_t i = 0; i < edges.n_cols; ++i)
        {
            size_t src = edges(0, i);
            size_t dst = edges(1, i);
            
            if (src < numNodes && dst < numNodes)
            {
                adjacencyList[src].push_back(dst);
                adjacencyList[dst].push_back(src); // Undirected graph
            }
        }
        
        // Compute node degrees
        degrees.set_size(numNodes);
        for (size_t i = 0; i < numNodes; ++i)
        {
            degrees(i) = adjacencyList[i].size();
        }
    }

    /**
     * Get neighbors of a node
     */
    const std::vector<size_t>& GetNeighbors(size_t nodeId) const
    {
        return adjacencyList[nodeId];
    }

    /**
     * Get subgraph around a node with k-hop neighborhood
     */
    void GetKHopNeighborhood(size_t nodeId, size_t k, 
                            uvec& neighborhoodNodes,
                            umat& neighborhoodEdges) const
    {
        std::vector<size_t> nodes;
        std::unordered_map<size_t, size_t> nodeMap;
        std::vector<std::pair<size_t, size_t>> edges;
        
        std::queue<std::pair<size_t, size_t>> q; // (node, distance)
        std::unordered_set<size_t> visited;
        
        q.push({nodeId, 0});
        visited.insert(nodeId);
        
        while (!q.empty())
        {
            auto [currentNode, distance] = q.front();
            q.pop();
            
            nodes.push_back(currentNode);
            nodeMap[currentNode] = nodes.size() - 1;
            
            if (distance < k)
            {
                for (size_t neighbor : GetNeighbors(currentNode))
                {
                    if (visited.find(neighbor) == visited.end())
                    {
                        visited.insert(neighbor);
                        q.push({neighbor, distance + 1});
                    }
                    
                    // Add edge in both directions for undirected graph
                    edges.push_back({currentNode, neighbor});
                }
            }
        }
        
        // Convert to armadillo matrices
        neighborhoodNodes.set_size(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            neighborhoodNodes(i) = nodes[i];
        }
        
        neighborhoodEdges.set_size(2, edges.size());
        for (size_t i = 0; i < edges.size(); ++i)
        {
            neighborhoodEdges(0, i) = nodeMap[edges[i].first];
            neighborhoodEdges(1, i) = nodeMap[edges[i].second];
        }
    }

    /**
     * Get batch of nodes for training
     */
    void GetBatch(const uvec& batchNodes,
                  mat& batchFeatures,
                  umat& batchEdges,
                  uvec& batchLabels) const
    {
        batchFeatures = nodeFeatures.cols(batchNodes);
        
        if (labels.n_elem > 0)
        {
            batchLabels.set_size(batchNodes.n_elem);
            for (size_t i = 0; i < batchNodes.n_elem; ++i)
            {
                batchLabels(i) = labels(batchNodes(i));
            }
        }
        
        // For simplicity, return full graph edges
        // In practice, you might want to extract subgraph edges
        batchEdges = edges;
    }

    // Getters
    const mat& GetNodeFeatures() const { return nodeFeatures; }
    const umat& GetEdges() const { return edges; }
    const uvec& GetLabels() const { return labels; }
    size_t GetNumNodes() const { return numNodes; }
    size_t GetFeatureDim() const { return featureDim; }

private:
    mat nodeFeatures;    // (feature_dim, num_nodes)
    umat edges;          // (2, num_edges)
    uvec labels;         // (num_nodes,) - node classes
    size_t numNodes;
    size_t featureDim;
    std::vector<std::vector<size_t>> adjacencyList;
    uvec degrees;
};

/**
 * Graph Convolution Layer with Message Passing
 * Implements GCN-style message passing
 */
template<typename MatType = arma::mat>
class GraphConvLayer : public Layer<MatType>
{
public:
    GraphConvLayer(const size_t inputDim,
                   const size_t outputDim,
                   const bool useBias = true,
                   const std::string& aggregation = "mean") :
        Layer<MatType>(),
        inputDim(inputDim),
        outputDim(outputDim),
        useBias(useBias),
        aggregation(aggregation)
    {
        InitializeWeights();
    }

    void InitializeWeights()
    {
        // Self weight matrix
        W_self.set_size(inputDim, outputDim);
        HeInitialization().Initialize(W_self, inputDim, outputDim);
        
        // Neighbor weight matrix
        W_neigh.set_size(inputDim, outputDim);
        HeInitialization().Initialize(W_neigh, inputDim, outputDim);
        
        // Bias
        if (useBias)
        {
            bias.set_size(outputDim, 1);
            bias.zeros();
        }
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input: (inputDim * numNodes, batchSize)
        // For GNNs, we process the entire graph, so batchSize is typically 1
        const size_t batchSize = input.n_cols;
        const size_t numNodes = input.n_rows / inputDim;
        
        output.set_size(outputDim * numNodes, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType nodeFeatures = input.col(b);
            nodeFeatures.reshape(inputDim, numNodes);
            
            MatType transformedOutput(outputDim, numNodes);
            
            // Apply self transformation
            MatType selfFeatures = W_self.t() * nodeFeatures;
            
            // Apply neighbor aggregation
            MatType neighborFeatures = AggregateNeighbors(nodeFeatures);
            MatType neighFeatures = W_neigh.t() * neighborFeatures;
            
            // Combine self and neighbor information
            transformedOutput = selfFeatures + neighFeatures;
            
            if (useBias)
            {
                transformedOutput.each_col() += bias;
            }
            
            // Apply activation function
            transformedOutput = arma::max(transformedOutput, 0.0); // ReLU
            
            output.col(b) = arma::vectorise(transformedOutput);
        }
        
        // Store for backward pass
        previousInput = input;
        previousOutput = output;
    }

    MatType AggregateNeighbors(const MatType& nodeFeatures)
    {
        const size_t numNodes = nodeFeatures.n_cols;
        MatType aggregated(outputDim, numNodes);
        aggregated.zeros();
        
        // This is a simplified aggregation
        // In practice, you would use the actual graph structure
        // For mean aggregation
        if (aggregation == "mean")
        {
            // Simple mean of all nodes (placeholder)
            // Real implementation would use adjacency matrix
            for (size_t i = 0; i < numNodes; ++i)
            {
                vec neighborSum = arma::zeros<vec>(inputDim);
                size_t neighborCount = 0;
                
                // In real implementation, iterate over actual neighbors
                // For now, use all nodes as neighbors (simplified)
                for (size_t j = 0; j < numNodes; ++j)
                {
                    if (i != j)
                    {
                        neighborSum += nodeFeatures.col(j);
                        neighborCount++;
                    }
                }
                
                if (neighborCount > 0)
                {
                    aggregated.col(i) = neighborSum / neighborCount;
                }
            }
        }
        else if (aggregation == "sum")
        {
            // Sum aggregation
            for (size_t i = 0; i < numNodes; ++i)
            {
                vec neighborSum = arma::zeros<vec>(inputDim);
                
                for (size_t j = 0; j < numNodes; ++j)
                {
                    if (i != j)
                    {
                        neighborSum += nodeFeatures.col(j);
                    }
                }
                
                aggregated.col(i) = neighborSum;
            }
        }
        
        return aggregated;
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        // Simplified backward pass
        g.set_size(input.n_rows, input.n_cols);
        g = gy; // Placeholder implementation
        
        // In practice, compute proper gradients through message passing
    }

    virtual GraphConvLayer* Clone() const
    {
        return new GraphConvLayer(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inputDim));
        ar(CEREAL_NVP(outputDim));
        ar(CEREAL_NVP(useBias));
        ar(CEREAL_NVP(aggregation));
        ar(CEREAL_NVP(W_self));
        ar(CEREAL_NVP(W_neigh));
        ar(CEREAL_NVP(bias));
    }

private:
    size_t inputDim;
    size_t outputDim;
    bool useBias;
    std::string aggregation;
    
    MatType W_self;
    MatType W_neigh;
    MatType bias;
    MatType previousInput;
    MatType previousOutput;
};

/**
 * Graph Attention Layer
 * Implements GAT-style attention mechanism
 */
template<typename MatType = arma::mat>
class GraphAttentionLayer : public Layer<MatType>
{
public:
    GraphAttentionLayer(const size_t inputDim,
                        const size_t outputDim,
                        const size_t numHeads = 8,
                        const double dropoutRate = 0.6,
                        const double negativeSlope = 0.2) :
        Layer<MatType>(),
        inputDim(inputDim),
        outputDim(outputDim),
        numHeads(numHeads),
        dropoutRate(dropoutRate),
        negativeSlope(negativeSlope)
    {
        InitializeWeights();
    }

    void InitializeWeights()
    {
        // Weight matrix for each attention head
        W.set_size(inputDim, outputDim * numHeads);
        HeInitialization().Initialize(W, inputDim, outputDim * numHeads);
        
        // Attention parameters
        a.set_size(2 * outputDim, numHeads);
        HeInitialization().Initialize(a, 2 * outputDim, numHeads);
        
        // Dropout
        if (dropoutRate > 0.0)
        {
            dropout = std::make_unique<Dropout<>>(dropoutRate);
        }
    }

    void Forward(const MatType& input, MatType& output) override
    {
        const size_t batchSize = input.n_cols;
        const size_t numNodes = input.n_rows / inputDim;
        
        output.set_size(outputDim * numHeads * numNodes, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType nodeFeatures = input.col(b);
            nodeFeatures.reshape(inputDim, numNodes);
            
            // Linear transformation
            MatType transformed = W.t() * nodeFeatures; // (outputDim * numHeads, numNodes)
            transformed.reshape(outputDim, numHeads * numNodes);
            
            // Apply attention mechanism
            MatType attended = ApplyAttention(transformed, numNodes);
            
            // Apply dropout
            if (dropoutRate > 0.0)
            {
                dropout->Forward(attended, attended);
            }
            
            // Apply LeakyReLU
            attended.transform([this](double val) { 
                return val > 0 ? val : negativeSlope * val; 
            });
            
            output.col(b) = arma::vectorise(attended);
        }
    }

    MatType ApplyAttention(const MatType& features, size_t numNodes)
    {
        MatType output(outputDim * numHeads, numNodes);
        
        for (size_t head = 0; head < numHeads; ++head)
        {
            MatType headFeatures = features.rows(head * outputDim, 
                                               (head + 1) * outputDim - 1);
            headFeatures.reshape(outputDim, numNodes);
            
            // Compute attention scores
            MatType attentionScores = ComputeAttentionScores(headFeatures, numNodes);
            
            // Apply attention to features
            for (size_t i = 0; i < numNodes; ++i)
            {
                vec attendedFeature = arma::zeros<vec>(outputDim);
                
                for (size_t j = 0; j < numNodes; ++j)
                {
                    attendedFeature += attentionScores(i, j) * headFeatures.col(j);
                }
                
                output.rows(head * outputDim + i * outputDim * numHeads,
                          head * outputDim + (i + 1) * outputDim * numHeads - 1) = 
                    attendedFeature;
            }
        }
        
        return output;
    }

    MatType ComputeAttentionScores(const MatType& features, size_t numNodes)
    {
        MatType scores(numNodes, numNodes);
        
        // Compute attention coefficients
        for (size_t i = 0; i < numNodes; ++i)
        {
            for (size_t j = 0; j < numNodes; ++j)
            {
                // Concatenate features
                vec concat = arma::join_vert(features.col(i), features.col(j));
                
                // Compute attention score: a^T [Wh_i || Wh_j]
                double score = arma::dot(a.col(0), concat);
                
                // Apply LeakyReLU
                score = score > 0 ? score : negativeSlope * score;
                
                scores(i, j) = std::exp(score);
            }
        }
        
        // Apply softmax normalization
        for (size_t i = 0; i < numNodes; ++i)
        {
            scores.row(i) = scores.row(i) / arma::sum(scores.row(i));
        }
        
        return scores;
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        g = gy; // Simplified backward pass
    }

    virtual GraphAttentionLayer* Clone() const
    {
        return new GraphAttentionLayer(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inputDim));
        ar(CEREAL_NVP(outputDim));
        ar(CEREAL_NVP(numHeads));
        ar(CEREAL_NVP(dropoutRate));
        ar(CEREAL_NVP(negativeSlope));
        ar(CEREAL_NVP(W));
        ar(CEREAL_NVP(a));
        ar(CEREAL_NVP(dropout));
    }

private:
    size_t inputDim;
    size_t outputDim;
    size_t numHeads;
    double dropoutRate;
    double negativeSlope;
    
    MatType W;
    MatType a;
    std::unique_ptr<Dropout<>> dropout;
};

/**
 * Attention Pooling Layer for Graph Classification
 * Aggregates node features into graph-level representation
 */
template<typename MatType = arma::mat>
class AttentionPooling : public Layer<MatType>
{
public:
    AttentionPooling(const size_t inputDim,
                     const size_t outputDim) :
        Layer<MatType>(),
        inputDim(inputDim),
        outputDim(outputDim)
    {
        InitializeWeights();
    }

    void InitializeWeights()
    {
        // Attention weights
        W_att.set_size(inputDim, outputDim);
        HeInitialization().Initialize(W_att, inputDim, outputDim);
        
        // Context vector for attention
        context.set_size(outputDim, 1);
        HeInitialization().Initialize(context, outputDim, 1);
    }

    void Forward(const MatType& input, MatType& output) override
    {
        // Input: (inputDim * numNodes, batchSize)
        const size_t batchSize = input.n_cols;
        const size_t numNodes = input.n_rows / inputDim;
        
        output.set_size(outputDim, batchSize);
        
        for (size_t b = 0; b < batchSize; ++b)
        {
            MatType nodeFeatures = input.col(b);
            nodeFeatures.reshape(inputDim, numNodes);
            
            // Compute attention scores for each node
            vec attentionScores(numNodes);
            
            for (size_t i = 0; i < numNodes; ++i)
            {
                // Transform node features
                vec transformed = W_att.t() * nodeFeatures.col(i);
                
                // Compute attention score: tanh(context^T * transformed)
                double score = std::tanh(arma::dot(context, transformed));
                attentionScores(i) = score;
            }
            
            // Apply softmax to get attention weights
            attentionScores = arma::exp(attentionScores);
            attentionScores = attentionScores / arma::sum(attentionScores);
            
            // Compute weighted sum of node features
            vec graphRepresentation(outputDim, arma::fill::zeros);
            
            for (size_t i = 0; i < numNodes; ++i)
            {
                vec transformed = W_att.t() * nodeFeatures.col(i);
                graphRepresentation += attentionScores(i) * transformed;
            }
            
            output.col(b) = graphRepresentation;
        }
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        g = gy; // Simplified backward pass
    }

    virtual AttentionPooling* Clone() const
    {
        return new AttentionPooling(*this);
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(inputDim));
        ar(CEREAL_NVP(outputDim));
        ar(CEREAL_NVP(W_att));
        ar(CEREAL_NVP(context));
    }

private:
    size_t inputDim;
    size_t outputDim;
    MatType W_att;
    MatType context;
};

/**
 * Complete Graph Neural Network for Node Classification
 */
class GraphNeuralNetwork
{
public:
    GraphNeuralNetwork(const size_t inputDim,
                       const size_t hiddenDim,
                       const size_t outputDim,
                       const size_t numLayers = 3,
                       const std::string& layerType = "GAT") :
        inputDim(inputDim),
        hiddenDim(hiddenDim),
        outputDim(outputDim),
        numLayers(numLayers),
        layerType(layerType)
    {
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        network = std::make_unique<FFN<CrossEntropyError<>, HeInitialization>>();
        
        std::cout << "Building Graph Neural Network..." << std::endl;
        std::cout << "Input dimension: " << inputDim << std::endl;
        std::cout << "Hidden dimension: " << hiddenDim << std::endl;
        std::cout << "Output dimension: " << outputDim << std::endl;
        std::cout << "Number of layers: " << numLayers << std::endl;
        std::cout << "Layer type: " << layerType << std::endl;
        
        // Input layer
        if (layerType == "GCN")
        {
            network->Add<GraphConvLayer<>>(inputDim, hiddenDim, true, "mean");
            network->Add<ReLULayer<>>();
            network->Add<Dropout<>>(0.5);
            
            // Hidden layers
            for (size_t i = 1; i < numLayers - 1; ++i)
            {
                network->Add<GraphConvLayer<>>(hiddenDim, hiddenDim, true, "mean");
                network->Add<ReLULayer<>>();
                network->Add<Dropout<>>(0.5);
            }
            
            // Output layer
            network->Add<GraphConvLayer<>>(hiddenDim, outputDim, true, "mean");
        }
        else if (layerType == "GAT")
        {
            network->Add<GraphAttentionLayer<>>(inputDim, hiddenDim, 8, 0.6, 0.2);
            network->Add<ELU<>>();
            network->Add<Dropout<>>(0.6);
            
            // Hidden layers
            for (size_t i = 1; i < numLayers - 1; ++i)
            {
                network->Add<GraphAttentionLayer<>>(hiddenDim, hiddenDim, 8, 0.6, 0.2);
                network->Add<ELU<>>();
                network->Add<Dropout<>>(0.6);
            }
            
            // Output layer (single head for classification)
            network->Add<GraphAttentionLayer<>>(hiddenDim, outputDim, 1, 0.6, 0.2);
        }
        
        network->Add<LogSoftMax<>>();
        
        std::cout << "GNN architecture built successfully!" << std::endl;
    }

    /**
     * Train the GNN on graph data
     */
    void Train(const GraphData& graph,
               const size_t epochs = 100,
               const size_t batchSize = 32,
               const double learningRate = 0.001)
    {
        std::cout << "\nTraining Graph Neural Network..." << std::endl;
        std::cout << "Nodes: " << graph.GetNumNodes() << std::endl;
        std::cout << "Features: " << graph.GetFeatureDim() << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        
        ens::Adam optimizer(learningRate, batchSize, 0.9, 0.999, 1e-8);
        
        // Generate training batches (node indices)
        uvec allNodes = arma::regspace<uvec>(0, graph.GetNumNodes() - 1);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            double epochLoss = 0.0;
            size_t numBatches = 0;
            
            // Shuffle nodes
            allNodes = arma::shuffle(allNodes);
            
            for (size_t i = 0; i < allNodes.n_elem; i += batchSize)
            {
                size_t currentBatchSize = std::min(batchSize, allNodes.n_elem - i);
                uvec batchNodes = allNodes.subvec(i, i + currentBatchSize - 1);
                
                // Get batch data
                mat batchFeatures;
                umat batchEdges;
                uvec batchLabels;
                graph.GetBatch(batchNodes, batchFeatures, batchEdges, batchLabels);
                
                // Convert labels to one-hot encoding
                mat oneHotLabels = ConvertToOneHot(batchLabels, outputDim);
                
                // Forward pass and compute loss
                double loss = network->Evaluate(batchFeatures, oneHotLabels);
                network->Gradient(batchFeatures, oneHotLabels);
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
     * Predict node classes
     */
    mat Predict(const mat& nodeFeatures)
    {
        mat predictions;
        network->Predict(nodeFeatures, predictions);
        return predictions;
    }

    /**
     * Evaluate model on test nodes
     */
    double Evaluate(const GraphData& graph, const uvec& testNodes)
    {
        size_t correct = 0;
        
        for (size_t i = 0; i < testNodes.n_elem; ++i)
        {
            size_t nodeId = testNodes(i);
            mat nodeFeatures = graph.GetNodeFeatures().col(nodeId);
            mat prediction = Predict(nodeFeatures);
            
            uword predictedClass = prediction.index_max();
            uword trueClass = graph.GetLabels()(nodeId);
            
            if (predictedClass == trueClass)
                correct++;
        }
        
        return (static_cast<double>(correct) / testNodes.n_elem) * 100.0;
    }

    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "gnn_model", *network);
        std::cout << "GNN model saved to: " << filename << std::endl;
    }

    void LoadModel(const std::string& filename)
    {
        data::Load(filename, "gnn_model", *network);
        std::cout << "GNN model loaded from: " << filename << std::endl;
    }

private:
    size_t inputDim;
    size_t hiddenDim;
    size_t outputDim;
    size_t numLayers;
    std::string layerType;
    
    std::unique_ptr<FFN<CrossEntropyError<>, HeInitialization>> network;

    mat ConvertToOneHot(const uvec& labels, size_t numClasses)
    {
        mat oneHot(numClasses, labels.n_elem, arma::fill::zeros);
        
        for (size_t i = 0; i < labels.n_elem; ++i)
        {
            if (labels(i) < numClasses)
            {
                oneHot(labels(i), i) = 1.0;
            }
        }
        
        return oneHot;
    }
};

/**
 * Example: Cora Dataset Node Classification
 */
class CoraDataset
{
public:
    static GraphData LoadCoraDataset()
    {
        std::cout << "Loading Cora dataset (synthetic)..." << std::endl;
        
        // Cora dataset parameters
        const size_t numNodes = 2708;
        const size_t featureDim = 1433;
        const size_t numClasses = 7;
        
        // Generate synthetic node features
        mat nodeFeatures = arma::randu<mat>(featureDim, numNodes);
        
        // Generate synthetic edges (sparse connectivity)
        std::vector<std::pair<size_t, size_t>> edgeList;
        for (size_t i = 0; i < numNodes; ++i)
        {
            // Each node connects to 2-5 random neighbors
            size_t numEdges = 2 + (arma::randi() % 4);
            for (size_t j = 0; j < numEdges; ++j)
            {
                size_t neighbor = arma::randi() % numNodes;
                if (neighbor != i)
                {
                    edgeList.push_back({i, neighbor});
                }
            }
        }
        
        // Convert to adjacency matrix format
        umat edges(2, edgeList.size());
        for (size_t i = 0; i < edgeList.size(); ++i)
        {
            edges(0, i) = edgeList[i].first;
            edges(1, i) = edgeList[i].second;
        }
        
        // Generate synthetic labels
        uvec labels(numNodes);
        for (size_t i = 0; i < numNodes; ++i)
        {
            labels(i) = arma::randi() % numClasses;
        }
        
        std::cout << "Generated synthetic Cora dataset:" << std::endl;
        std::cout << "  Nodes: " << numNodes << std::endl;
        std::cout << "  Features: " << featureDim << std::endl;
        std::cout << "  Edges: " << edges.n_cols << std::endl;
        std::cout << "  Classes: " << numClasses << std::endl;
        
        return GraphData(nodeFeatures, edges, labels);
    }
};

/**
 * Main example: Node classification on Cora dataset
 */
int main()
{
    std::cout << "Graph Neural Network with Message Passing and Attention" << std::endl;
    std::cout << "======================================================" << std::endl;
    
    // Load dataset
    GraphData coraData = CoraDataset::LoadCoraDataset();
    
    // Create GNN model
    GraphNeuralNetwork gnn(coraData.GetFeatureDim(), 64, 7, 3, "GAT");
    
    // Split data into train/validation
    uvec allNodes = arma::regspace<uvec>(0, coraData.GetNumNodes() - 1);
    allNodes = arma::shuffle(allNodes);
    
    size_t trainSize = coraData.GetNumNodes() * 0.8;
    uvec trainNodes = allNodes.subvec(0, trainSize - 1);
    uvec testNodes = allNodes.subvec(trainSize, allNodes.n_elem - 1);
    
    std::cout << "Training nodes: " << trainNodes.n_elem << std::endl;
    std::cout << "Test nodes: " << testNodes.n_elem << std::endl;
    
    // Train the model
    gnn.Train(coraData, 50, 32, 0.001);
    
    // Evaluate on test set
    double testAccuracy = gnn.Evaluate(coraData, testNodes);
    std::cout << "\nTest Accuracy: " << testAccuracy << "%" << std::endl;
    
    // Save the model
    gnn.SaveModel("gnn_cora_model.xml");
    
    return 0;
}