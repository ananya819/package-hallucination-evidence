#include "EBGAN.hpp"
#include <mlpack/methods/ann/init_rules/gaussian_init.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <random>
#include <cmath>

EBGAN::EBGAN(size_t noiseDim,
             size_t generatorDim,
             size_t discriminatorDim,
             size_t dataDim,
             double learningRate,
             double margin,
             double pullawayWeight)
    : noiseDim(noiseDim), generatorDim(generatorDim),
      discriminatorDim(discriminatorDim), dataDim(dataDim),
      learningRate(learningRate), margin(margin),
      pullawayWeight(pullawayWeight) {

    // Build networks
    BuildGenerator();
    BuildDiscriminator();
    BuildAutoencoder();

    // Initialize optimizers
    generatorOptimizer = std::make_unique<Adam>(learningRate, 32, 0.5, 0.999, 1e-8);
    discriminatorOptimizer = std::make_unique<Adam>(learningRate, 32, 0.5, 0.999, 1e-8);

    InitializeNetworks();
}

void EBGAN::BuildGenerator() {
    // Generator: noise -> data space
    generator.Add<Linear>(256);
    generator.Add<BatchNorm<> >();
    generator.Add<ReLU>();
    generator.Add<Linear>(512);
    generator.Add<BatchNorm<> >();
    generator.Add<ReLU>();
    generator.Add<Linear>(1024);
    generator.Add<BatchNorm<> >();
    generator.Add<ReLU>();
    generator.Add<Linear>(dataDim);
    generator.Add<Tanh>(); // Output in [-1, 1]
}

void EBGAN::BuildDiscriminator() {
    // Discriminator feature extractor
    discriminator.Add<Linear>(1024);
    discriminator.Add<LeakyReLU>(0.2);
    discriminator.Add<Linear>(512);
    discriminator.Add<LeakyReLU>(0.2);
    discriminator.Add<Linear>(256);
    discriminator.Add<LeakyReLU>(0.2);
    discriminator.Add<Linear>(discriminatorDim); // Feature representation
}

void EBGAN::BuildAutoencoder() {
    // Autoencoder for discriminator
    // Encoder
    autoencoder.Add<Linear>(1024);
    autoencoder.Add<LeakyReLU>(0.2);
    autoencoder.Add<Linear>(512);
    autoencoder.Add<LeakyReLU>(0.2);
    
    // Bottleneck (features)
    autoencoder.Add<Linear>(discriminatorDim);
    autoencoder.Add<LeakyReLU>(0.2);
    
    // Decoder
    autoencoder.Add<Linear>(512);
    autoencoder.Add<LeakyReLU>(0.2);
    autoencoder.Add<Linear>(1024);
    autoencoder.Add<LeakyReLU>(0.2);
    autoencoder.Add<Linear>(dataDim);
    autoencoder.Add<Tanh>();
}

void EBGAN::InitializeNetworks() {
    // Initialize with specific ranges for better training
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // He initialization for ReLU networks
    auto heInit = [](size_t rows, size_t cols) {
        double stddev = std::sqrt(2.0 / rows);
        return arma::randn<arma::mat>(rows, cols) * stddev;
    };
    
    // Initialize generator
    generator.Parameters() = heInit(generator.Parameters().n_rows, 1);
    
    // Initialize discriminator and autoencoder
    discriminator.Parameters() = heInit(discriminator.Parameters().n_rows, 1);
    autoencoder.Parameters() = heInit(autoencoder.Parameters().n_rows, 1);
}

void EBGAN::Train(const arma::mat& realData) {
    const size_t batchSize = realData.n_cols;
    
    // Generate fake data
    arma::mat noise = arma::randn<arma::mat>(noiseDim, batchSize);
    arma::mat fakeData;
    generator.Predict(noise, fakeData);
    
    // Compute energies
    arma::vec realEnergy = ComputeEnergy(realData);
    arma::vec fakeEnergy = ComputeEnergy(fakeData);
    
    // Discriminator loss (hinge version)
    double discLoss = arma::mean(realEnergy) + 
                     arma::mean(arma::max(arma::zeros<arma::vec>(batchSize), 
                                         margin - fakeEnergy));
    
    // Generator loss
    double genLoss = arma::mean(fakeEnergy);
    
    // Add pull-away term for diversity
    double pullAway = ComputePullAwayTerm(fakeData);
    genLoss += pullawayWeight * pullAway;
    
    // Store losses
    discriminatorLoss.push_back(discLoss);
    generatorLoss.push_back(genLoss);
    energyScores.push_back(arma::mean(realEnergy));
    
    // Training step would go here - in practice you'd compute gradients
    // and update weights. This is simplified for demonstration.
}

arma::vec EBGAN::ComputeEnergy(const arma::mat& data) {
    // Energy is reconstruction error of autoencoder
    arma::mat reconstructed;
    autoencoder.Predict(data, reconstructed);
    
    // Compute MSE for each sample
    arma::vec energy(data.n_cols);
    for (size_t i = 0; i < data.n_cols; ++i) {
        energy(i) = arma::accu(arma::square(data.col(i) - reconstructed.col(i))) / data.n_rows;
    }
    
    return energy;
}

double EBGAN::ComputePullAwayTerm(const arma::mat& samples) {
    // Pull-away term to encourage sample diversity
    size_t batchSize = samples.n_cols;
    
    if (batchSize < 2) return 0.0;
    
    // Normalize samples
    arma::mat normalized = arma::normalise(samples, 2, 0);
    
    // Compute similarity matrix
    arma::mat similarity = normalized.t() * normalized;
    
    // Remove diagonal
    similarity.diag().zeros();
    
    // Sum of squared similarities
    double pt = arma::accu(arma::square(similarity)) / (batchSize * (batchSize - 1));
    
    return pt;
}

arma::mat EBGAN::Generate(size_t numSamples) {
    arma::mat noise = arma::randn<arma::mat>(noiseDim, numSamples);
    arma::mat generated;
    generator.Predict(noise, generated);
    return generated;
}

arma::mat EBGAN::Reconstruct(const arma::mat& data) {
    arma::mat reconstructed;
    autoencoder.Predict(data, reconstructed);
    return reconstructed;
}

arma::mat EBGAN::GetFeatures(const arma::mat& data) {
    // Get features from discriminator (encoder part)
    arma::mat features;
    
    // Forward pass through encoder part of autoencoder
    // For simplicity, we'll use the discriminator network
    discriminator.Predict(data, features);
    
    return features;
}

void EBGAN::Save(const std::string& path) {
    generator.Parameters().save(path + "_generator.bin");
    discriminator.Parameters().save(path + "_discriminator.bin");
    autoencoder.Parameters().save(path + "_autoencoder.bin");
}

void EBGAN::Load(const std::string& path) {
    arma::mat params;
    
    if (params.load(path + "_generator.bin")) {
        generator.Parameters() = params;
    }
    if (params.load(path + "_discriminator.bin")) {
        discriminator.Parameters() = params;
    }
    if (params.load(path + "_autoencoder.bin")) {
        autoencoder.Parameters() = params;
    }
}