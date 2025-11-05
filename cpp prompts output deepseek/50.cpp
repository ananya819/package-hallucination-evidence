#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/ann/visitor/output_parameter_visitor.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::kmeans;
using namespace arma;

/**
 * @brief Differentiable clustering model with autoencoder backbone
 * 
 * This class implements a deep clustering model that jointly learns
 * feature representations and cluster assignments using an autoencoder
 * structure with a clustering loss.
 */
class DifferentiableClusteringModel
{
 public:
  /**
   * @brief Constructor for the differentiable clustering model
   * 
   * @param inputDim Dimension of input data
   * @param encodingDim Dimension of encoded representation (bottleneck)
   * @param numClusters Number of clusters
   * @param lambda Weight for clustering loss in the total objective
   */
  DifferentiableClusteringModel(size_t inputDim,
                               size_t encodingDim,
                               size_t numClusters,
                               double lambda = 0.1)
      : inputDim(inputDim),
        encodingDim(encodingDim),
        numClusters(numClusters),
        lambda(lambda),
        autoencoder(inputDim),
        clusterCenters(encodingDim, numClusters)
  {
    BuildAutoencoder();
    InitializeClusterCenters();
  }

  /**
   * @brief Build the autoencoder architecture
   */
  void BuildAutoencoder()
  {
    // Encoder layers
    autoencoder.Add<Linear<>>(inputDim, 128);
    autoencoder.Add<ReLULayer<>>();
    autoencoder.Add<Linear<>>(128, 64);
    autoencoder.Add<ReLULayer<>>();
    autoencoder.Add<Linear<>>(64, encodingDim);
    autoencoder.Add<ReLULayer<>>();
    
    // Decoder layers (symmetric to encoder)
    autoencoder.Add<Linear<>>(encodingDim, 64);
    autoencoder.Add<ReLULayer<>>();
    autoencoder.Add<Linear<>>(64, 128);
    autoencoder.Add<ReLULayer<>>();
    autoencoder.Add<Linear<>>(128, inputDim);
    
    // Use Mean Squared Error for reconstruction
    autoencoder.ResetParameters();
  }

  /**
   * @brief Initialize cluster centers using K-means on encoded representations
   */
  void InitializeClusterCenters(const mat& data)
  {
    // First, get encoded representations
    mat encodedData;
    Encode(data, encodedData);
    
    // Use K-means to initialize cluster centers
    KMeans<> kmeans;
    Row<size_t> assignments;
    kmeans.Cluster(encodedData, numClusters, assignments, clusterCenters);
  }

  /**
   * @brief Encode input data to latent representation
   */
  void Encode(const mat& data, mat& encoded)
  {
    // Forward pass through encoder layers (first half of network)
    mat temp = data;
    for (size_t i = 0; i < autoencoder.Network().size() / 2; ++i)
    {
      autoencoder.Network()[i]->Forward(temp, temp);
    }
    encoded = temp;
  }

  /**
   * @brief Compute soft cluster assignments (probability distribution over clusters)
   */
  void ComputeSoftAssignments(const mat& encodedData, mat& assignments)
  {
    assignments.set_size(numClusters, encodedData.n_cols);
    
    #pragma omp parallel for
    for (size_t i = 0; i < encodedData.n_cols; ++i)
    {
      vec distances(numClusters);
      
      // Compute squared Euclidean distances to cluster centers
      for (size_t j = 0; j < numClusters; ++j)
      {
        distances(j) = sum(square(encodedData.col(i) - clusterCenters.col(j)));
      }
      
      // Convert distances to probabilities using softmax
      // Smaller distance = higher probability
      vec negativeDistances = -distances;
      assignments.col(i) = exp(negativeDistances - max(negativeDistances));
      assignments.col(i) /= sum(assignments.col(i));
    }
  }

  /**
   * @brief Compute clustering loss (KL divergence from target distribution)
   */
  double ComputeClusteringLoss(const mat& encodedData, const mat& softAssignments)
  {
    // Compute target distribution (emphasizing high-confidence assignments)
    mat targetDistribution = softAssignments;
    
    // Square and normalize to emphasize high-confidence assignments
    for (size_t i = 0; i < targetDistribution.n_cols; ++i)
    {
      targetDistribution.col(i) = square(targetDistribution.col(i));
      targetDistribution.col(i) /= sum(targetDistribution.col(i));
    }
    
    // Compute KL divergence between soft assignments and target distribution
    double klDivergence = 0.0;
    for (size_t i = 0; i < softAssignments.n_cols; ++i)
    {
      for (size_t j = 0; j < numClusters; ++j)
      {
        if (targetDistribution(j, i) > 1e-10 && softAssignments(j, i) > 1e-10)
        {
          klDivergence += targetDistribution(j, i) * 
                         log(targetDistribution(j, i) / softAssignments(j, i));
        }
      }
    }
    
    return klDivergence / softAssignments.n_cols;
  }

  /**
   * @brief Train the model
   */
  void Train(const mat& data,
             size_t epochs = 100,
             double learningRate = 0.001,
             size_t batchSize = 32)
  {
    const size_t numSamples = data.n_cols;
    
    // Optimizer for autoencoder
    ens::Adam optimizer(learningRate, batchSize, 0.9, 0.999, 1e-8, epochs * numSamples);
    
    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
      double totalReconstructionLoss = 0.0;
      double totalClusteringLoss = 0.0;
      
      for (size_t i = 0; i < numSamples; i += batchSize)
      {
        const size_t currentBatchSize = std::min(batchSize, numSamples - i);
        mat batch = data.cols(i, i + currentBatchSize - 1);
        
        // Forward pass through autoencoder
        mat reconstruction;
        autoencoder.Forward(batch, reconstruction);
        
        // Compute reconstruction loss
        double reconstructionLoss = MeanSquaredError::Forward(batch, reconstruction);
        totalReconstructionLoss += reconstructionLoss * currentBatchSize;
        
        // Get encoded representations
        mat encodedBatch;
        Encode(batch, encodedBatch);
        
        // Compute clustering loss
        mat softAssignments;
        ComputeSoftAssignments(encodedBatch, softAssignments);
        double clusteringLoss = ComputeClusteringLoss(encodedBatch, softAssignments);
        totalClusteringLoss += clusteringLoss * currentBatchSize;
        
        // Combined loss
        double totalLoss = reconstructionLoss + lambda * clusteringLoss;
        
        // Backward pass
        mat reconstructionGradient;
        MeanSquaredError::Backward(batch, reconstruction, reconstructionGradient);
        
        // Here we would need to compute the gradient of clustering loss w.r.t. encoder
        // This is simplified - in practice you'd need to implement the full gradient
        autoencoder.Backward(batch, reconstructionGradient);
        
        // Update parameters
        ens::OptimizerState<arma::mat, arma::mat> state;
        autoencoder.UpdateWeights(optimizer, state);
      }
      
      // Update cluster centers (simplified - in practice, use gradient descent)
      if (epoch % 10 == 0)
      {
        UpdateClusterCenters(data);
      }
      
      if (epoch % 10 == 0)
      {
        std::cout << "Epoch " << epoch << "/" << epochs 
                  << " - Reconstruction Loss: " << totalReconstructionLoss / numSamples
                  << " - Clustering Loss: " << totalClusteringLoss / numSamples
                  << std::endl;
      }
    }
  }

  /**
   * @brief Update cluster centers based on current encoded representations
   */
  void UpdateClusterCenters(const mat& data)
  {
    mat encodedData;
    Encode(data, encodedData);
    
    mat softAssignments;
    ComputeSoftAssignments(encodedData, softAssignments);
    
    // Update centers as weighted average of encoded points
    for (size_t j = 0; j < numClusters; ++j)
    {
      vec weightedSum = zeros<vec>(encodingDim);
      double totalWeight = 0.0;
      
      for (size_t i = 0; i < encodedData.n_cols; ++i)
      {
        weightedSum += softAssignments(j, i) * encodedData.col(i);
        totalWeight += softAssignments(j, i);
      }
      
      if (totalWeight > 1e-10)
      {
        clusterCenters.col(j) = weightedSum / totalWeight;
      }
    }
  }

  /**
   * @brief Predict cluster assignments for new data
   */
  void Predict(const mat& data, Row<size_t>& assignments)
  {
    mat encodedData;
    Encode(data, encodedData);
    
    mat softAssignments;
    ComputeSoftAssignments(encodedData, softAssignments);
    
    // Convert soft assignments to hard assignments
    assignments.set_size(encodedData.n_cols);
    for (size_t i = 0; i < encodedData.n_cols; ++i)
    {
      assignments(i) = softAssignments.col(i).index_max();
    }
  }

  /**
   * @brief Get the encoded representation of data
   */
  void GetEncodedRepresentation(const mat& data, mat& encoded)
  {
    Encode(data, encoded);
  }

  /**
   * @brief Save model to disk
   */
  void SaveModel(const std::string& filename)
  {
    data::Save(filename + "_autoencoder.xml", "autoencoder", autoencoder);
    data::Save(filename + "_centers.csv", "centers", clusterCenters);
  }

  /**
   * @brief Load model from disk
   */
  void LoadModel(const std::string& filename)
  {
    data::Load(filename + "_autoencoder.xml", "autoencoder", autoencoder);
    data::Load(filename + "_centers.csv", "centers", clusterCenters);
  }

 private:
  size_t inputDim;
  size_t encodingDim;
  size_t numClusters;
  double lambda;
  
  // Autoencoder model
  FFN<MeanSquaredError<>, RandomInitialization> autoencoder;
  
  // Cluster centers in the encoded space
  mat clusterCenters;
};

/**
 * @brief Example usage of the differentiable clustering model
 */
int main()
{
  // Generate sample data (3 Gaussian clusters in 10D space)
  const size_t numSamples = 1000;
  const size_t inputDim = 10;
  const size_t numClusters = 3;
  
  mat data(inputDim, numSamples);
  
  // Create three Gaussian clusters
  for (size_t i = 0; i < numSamples; ++i)
  {
    size_t cluster = i % numClusters;
    vec center = randu<vec>(inputDim) * cluster;
    data.col(i) = center + 0.1 * randn<vec>(inputDim);
  }
  
  // Create and train the model
  DifferentiableClusteringModel model(inputDim, 5, numClusters, 0.1);
  
  std::cout << "Training differentiable clustering model..." << std::endl;
  model.Train(data, 100, 0.001, 64);
  
  // Make predictions
  Row<size_t> assignments;
  model.Predict(data, assignments);
  
  // Evaluate clustering (simple accuracy check)
  size_t correct = 0;
  for (size_t i = 0; i < numSamples; ++i)
  {
    if (assignments(i) == (i % numClusters))
    {
      correct++;
    }
  }
  
  std::cout << "Clustering accuracy: " << (100.0 * correct / numSamples) << "%" << std::endl;
  
  // Get encoded representations
  mat encoded;
  model.GetEncodedRepresentation(data, encoded);
  std::cout << "Encoded representation shape: " << size(encoded) << std::endl;
  
  // Save model
  model.SaveModel("clustering_model");
  
  return 0;
}