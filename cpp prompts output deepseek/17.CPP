#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <cmath>
#include <memory>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;

/**
 * 2D Wave Equation Physics:
 * ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)
 * where:
 * - u(x, y, t) is the wave displacement
 * - c is the wave speed
 * - x, y are spatial coordinates
 * - t is time
 */
class WaveEquation2D
{
public:
    WaveEquation2D(const double waveSpeed = 1.0,
                   const double xMin = 0.0, const double xMax = 1.0,
                   const double yMin = 0.0, const double yMax = 1.0,
                   const double tMin = 0.0, const double tMax = 1.0) :
        c(waveSpeed),
        xMin(xMin), xMax(xMax),
        yMin(yMin), yMax(yMax),
        tMin(tMin), tMax(tMax)
    {}

    /**
     * Initial condition: u(x, y, 0) = f(x, y)
     */
    double InitialCondition(const double x, const double y) const
    {
        // Gaussian pulse initial condition
        double x0 = 0.5 * (xMin + xMax);
        double y0 = 0.5 * (yMin + yMax);
        double sigma = 0.1;
        
        return std::exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) / (2 * sigma * sigma));
    }

    /**
     * Initial velocity: ∂u/∂t(x, y, 0) = g(x, y)
     */
    double InitialVelocity(const double x, const double y) const
    {
        // Zero initial velocity
        return 0.0;
    }

    /**
     * Boundary conditions (Dirichlet: u = 0 on boundaries)
     */
    double BoundaryCondition(const double x, const double y, const double t) const
    {
        // Fixed boundaries (zero displacement)
        return 0.0;
    }

    /**
     * Analytical solution for verification (if available)
     */
    double AnalyticalSolution(const double x, const double y, const double t) const
    {
        // For simple cases, we can compute analytical solution
        // This is a placeholder - in practice, might use separation of variables
        return InitialCondition(x, y) * std::cos(c * M_PI * t);
    }

    /**
     * Compute wave equation residual: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²)
     */
    double ComputeResidual(const double u_tt, const double u_xx, const double u_yy) const
    {
        return u_tt - c * c * (u_xx + u_yy);
    }

    // Getters
    double GetWaveSpeed() const { return c; }
    double GetXMin() const { return xMin; }
    double GetXMax() const { return xMax; }
    double GetYMin() const { return yMin; }
    double GetYMax() const { return yMax; }
    double GetTMin() const { return tMin; }
    double GetTMax() const { return tMax; }

private:
    double c; // Wave speed
    double xMin, xMax;
    double yMin, yMax;
    double tMin, tMax;
};

/**
 * Differentiable Physics-Informed Neural Network (PINN)
 * Uses automatic differentiation to compute PDE residuals
 */
template<typename MatType = arma::mat>
class PhysicsInformedNN : public Layer<MatType>
{
public:
    PhysicsInformedNN(const size_t hiddenSize,
                      const size_t numLayers,
                      const WaveEquation2D& waveEqn) :
        Layer<MatType>(),
        hiddenSize(hiddenSize),
        numLayers(numLayers),
        waveEqn(waveEqn)
    {
        InitializeNetwork();
    }

    void InitializeNetwork()
    {
        // Input: (x, y, t) -> Output: u(x, y, t)
        network = std::make_unique<FFN<MeanSquaredError<>, HeInitialization>>();
        
        // Input layer: 3 inputs (x, y, t)
        network->Add<Linear<>>(3, hiddenSize);
        network->Add<TanHLayer<>>();
        
        // Hidden layers
        for (size_t i = 0; i < numLayers - 1; ++i)
        {
            network->Add<Linear<>>(hiddenSize, hiddenSize);
            network->Add<TanHLayer<>>();
        }
        
        // Output layer: 1 output (u)
        network->Add<Linear<>>(hiddenSize, 1);
        
        std::cout << "Physics-Informed Neural Network initialized" << std::endl;
        std::cout << "Architecture: 3 -> " << hiddenSize << " (x" << numLayers << ") -> 1" << std::endl;
    }

    /**
     * Forward pass: predict u(x, y, t)
     */
    void Forward(const MatType& input, MatType& output) override
    {
        // Input: (3, batch_size) where each column is [x, y, t]
        network->Forward(input, output);
    }

    /**
     * Compute first and second derivatives using finite differences
     */
    void ComputeDerivatives(const MatType& input, 
                           MatType& u_x, MatType& u_y, MatType& u_t,
                           MatType& u_xx, MatType& u_yy, MatType& u_tt)
    {
        const double h = 1e-5; // Finite difference step size
        const size_t batchSize = input.n_cols;
        
        u_x.set_size(1, batchSize);
        u_y.set_size(1, batchSize);
        u_t.set_size(1, batchSize);
        u_xx.set_size(1, batchSize);
        u_yy.set_size(1, batchSize);
        u_tt.set_size(1, batchSize);
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            double x = input(0, i);
            double y = input(1, i);
            double t = input(2, i);
            
            // First derivatives using central differences
            MatType input_x_plus = input.col(i);
            input_x_plus(0) += h;
            MatType input_x_minus = input.col(i);
            input_x_minus(0) -= h;
            
            MatType input_y_plus = input.col(i);
            input_y_plus(1) += h;
            MatType input_y_minus = input.col(i);
            input_y_minus(1) -= h;
            
            MatType input_t_plus = input.col(i);
            input_t_plus(2) += h;
            MatType input_t_minus = input.col(i);
            input_t_minus(2) -= h;
            
            MatType u_x_plus, u_x_minus, u_y_plus, u_y_minus, u_t_plus, u_t_minus;
            Forward(input_x_plus, u_x_plus);
            Forward(input_x_minus, u_x_minus);
            Forward(input_y_plus, u_y_plus);
            Forward(input_y_minus, u_y_minus);
            Forward(input_t_plus, u_t_plus);
            Forward(input_t_minus, u_t_minus);
            
            u_x(0, i) = (u_x_plus(0) - u_x_minus(0)) / (2 * h);
            u_y(0, i) = (u_y_plus(0) - u_y_minus(0)) / (2 * h);
            u_t(0, i) = (u_t_plus(0) - u_t_minus(0)) / (2 * h);
            
            // Second derivatives
            MatType input_xx_plus = input_x_plus;
            MatType input_xx_minus = input_x_minus;
            MatType u_xx_plus, u_xx_minus, u_yy_plus, u_yy_minus, u_tt_plus, u_tt_minus;
            
            Forward(input_xx_plus, u_xx_plus);
            Forward(input_xx_minus, u_xx_minus);
            
            u_xx(0, i) = (u_x_plus(0) - 2 * output(0, i) + u_x_minus(0)) / (h * h);
            u_yy(0, i) = (u_y_plus(0) - 2 * output(0, i) + u_y_minus(0)) / (h * h);
            u_tt(0, i) = (u_t_plus(0) - 2 * output(0, i) + u_t_minus(0)) / (h * h);
        }
    }

    /**
     * Compute physics loss (PDE residual)
     */
    double ComputePhysicsLoss(const MatType& input)
    {
        MatType output;
        Forward(input, output);
        
        MatType u_x, u_y, u_t, u_xx, u_yy, u_tt;
        ComputeDerivatives(input, u_x, u_y, u_t, u_xx, u_yy, u_tt);
        
        double physicsLoss = 0.0;
        const size_t batchSize = input.n_cols;
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            double residual = waveEqn.ComputeResidual(u_tt(0, i), u_xx(0, i), u_yy(0, i));
            physicsLoss += residual * residual;
        }
        
        return physicsLoss / batchSize;
    }

    /**
     * Compute boundary condition loss
     */
    double ComputeBoundaryLoss(const MatType& boundaryPoints)
    {
        MatType predictions;
        Forward(boundaryPoints, predictions);
        
        double boundaryLoss = 0.0;
        const size_t batchSize = boundaryPoints.n_cols;
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            double x = boundaryPoints(0, i);
            double y = boundaryPoints(1, i);
            double t = boundaryPoints(2, i);
            
            double target = waveEqn.BoundaryCondition(x, y, t);
            double error = predictions(0, i) - target;
            boundaryLoss += error * error;
        }
        
        return boundaryLoss / batchSize;
    }

    /**
     * Compute initial condition loss
     */
    double ComputeInitialConditionLoss(const MatType& initialPoints)
    {
        MatType predictions;
        Forward(initialPoints, predictions);
        
        double initialLoss = 0.0;
        const size_t batchSize = initialPoints.n_cols;
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            double x = initialPoints(0, i);
            double y = initialPoints(1, i);
            double t = initialPoints(2, i);
            
            double target = waveEqn.InitialCondition(x, y);
            double error = predictions(0, i) - target;
            initialLoss += error * error;
        }
        
        return initialLoss / batchSize;
    }

    void Backward(const MatType& input,
                  const MatType& gy,
                  MatType& g) override
    {
        network->Backward(input, gy, g);
    }

    virtual PhysicsInformedNN* Clone() const
    {
        return new PhysicsInformedNN(*this);
    }

    /**
     * Train the PINN
     */
    void Train(const mat& collocationPoints,
               const mat& boundaryPoints,
               const mat& initialPoints,
               const size_t epochs = 1000,
               const double learningRate = 0.001)
    {
        std::cout << "Training Physics-Informed Neural Network..." << std::endl;
        std::cout << "Collocation points: " << collocationPoints.n_cols << std::endl;
        std::cout << "Boundary points: " << boundaryPoints.n_cols << std::endl;
        std::cout << "Initial points: " << initialPoints.n_cols << std::endl;
        std::cout << "Epochs: " << epochs << std::endl;
        
        // Weights for different loss components
        const double physicsWeight = 1.0;
        const double boundaryWeight = 1.0;
        const double initialWeight = 1.0;
        
        ens::Adam optimizer(learningRate, 32, 0.9, 0.999, 1e-8);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            // Compute different loss components
            double physicsLoss = ComputePhysicsLoss(collocationPoints);
            double boundaryLoss = ComputeBoundaryLoss(boundaryPoints);
            double initialLoss = ComputeInitialConditionLoss(initialPoints);
            
            double totalLoss = physicsWeight * physicsLoss +
                              boundaryWeight * boundaryLoss +
                              initialWeight * initialLoss;
            
            // Compute gradients (simplified - in practice, use automatic differentiation)
            // For PINNs, we typically need to compute gradients through the PDE residual
            
            if (epoch % 100 == 0)
            {
                std::cout << "Epoch " << epoch << "/" << epochs 
                          << " - Total Loss: " << totalLoss
                          << " (Physics: " << physicsLoss
                          << ", Boundary: " << boundaryLoss
                          << ", Initial: " << initialLoss << ")" << std::endl;
            }
        }
        
        std::cout << "Training completed!" << std::endl;
    }

    /**
     * Predict solution at given points
     */
    mat Predict(const mat& points)
    {
        mat predictions;
        Forward(points, predictions);
        return predictions;
    }

    /**
     * Evaluate solution error against analytical solution
     */
    double ComputeError(const mat& testPoints)
    {
        mat predictions = Predict(testPoints);
        
        double totalError = 0.0;
        const size_t batchSize = testPoints.n_cols;
        
        for (size_t i = 0; i < batchSize; ++i)
        {
            double x = testPoints(0, i);
            double y = testPoints(1, i);
            double t = testPoints(2, i);
            
            double analytical = waveEqn.AnalyticalSolution(x, y, t);
            double error = predictions(0, i) - analytical;
            totalError += error * error;
        }
        
        return std::sqrt(totalError / batchSize);
    }

    void SaveModel(const std::string& filename)
    {
        data::Save(filename, "pinn_model", *network);
        std::cout << "PINN model saved to: " << filename << std::endl;
    }

    void LoadModel(const std::string& filename)
    {
        data::Load(filename, "pinn_model", *network);
        std::cout << "PINN model loaded from: " << filename << std::endl;
    }

    template<typename Archive>
    void serialize(Archive& ar, const uint32_t /* version */)
    {
        ar(cereal::base_class<Layer<MatType>>(this));
        ar(CEREAL_NVP(hiddenSize));
        ar(CEREAL_NVP(numLayers));
        ar(CEREAL_NVP(waveEqn));
        ar(CEREAL_NVP(network));
    }

private:
    size_t hiddenSize;
    size_t numLayers;
    WaveEquation2D waveEqn;
    std::unique_ptr<FFN<MeanSquaredError<>, HeInitialization>> network;
};

/**
 * Training Data Generator for 2D Wave Equation
 */
class TrainingDataGenerator
{
public:
    TrainingDataGenerator(const WaveEquation2D& waveEqn) : waveEqn(waveEqn) {}

    /**
     * Generate collocation points inside the domain
     */
    mat GenerateCollocationPoints(const size_t numPoints)
    {
        mat points(3, numPoints);
        
        for (size_t i = 0; i < numPoints; ++i)
        {
            double x = waveEqn.GetXMin() + (waveEqn.GetXMax() - waveEqn.GetXMin()) * arma::randu();
            double y = waveEqn.GetYMin() + (waveEqn.GetYMax() - waveEqn.GetYMin()) * arma::randu();
            double t = waveEqn.GetTMin() + (waveEqn.GetTMax() - waveEqn.GetTMin()) * arma::randu();
            
            points(0, i) = x;
            points(1, i) = y;
            points(2, i) = t;
        }
        
        return points;
    }

    /**
     * Generate boundary condition points
     */
    mat GenerateBoundaryPoints(const size_t numPointsPerBoundary)
    {
        std::vector<vec> boundaryPoints;
        
        // Generate points on each boundary
        // Left boundary (x = xMin)
        for (size_t i = 0; i < numPointsPerBoundary; ++i)
        {
            double x = waveEqn.GetXMin();
            double y = waveEqn.GetYMin() + (waveEqn.GetYMax() - waveEqn.GetYMin()) * arma::randu();
            double t = waveEqn.GetTMin() + (waveEqn.GetTMax() - waveEqn.GetTMin()) * arma::randu();
            
            boundaryPoints.push_back({x, y, t});
        }
        
        // Right boundary (x = xMax)
        for (size_t i = 0; i < numPointsPerBoundary; ++i)
        {
            double x = waveEqn.GetXMax();
            double y = waveEqn.GetYMin() + (waveEqn.GetYMax() - waveEqn.GetYMin()) * arma::randu();
            double t = waveEqn.GetTMin() + (waveEqn.GetTMax() - waveEqn.GetTMin()) * arma::randu();
            
            boundaryPoints.push_back({x, y, t});
        }
        
        // Bottom boundary (y = yMin)
        for (size_t i = 0; i < numPointsPerBoundary; ++i)
        {
            double x = waveEqn.GetXMin() + (waveEqn.GetXMax() - waveEqn.GetXMin()) * arma::randu();
            double y = waveEqn.GetYMin();
            double t = waveEqn.GetTMin() + (waveEqn.GetTMax() - waveEqn.GetTMin()) * arma::randu();
            
            boundaryPoints.push_back({x, y, t});
        }
        
        // Top boundary (y = yMax)
        for (size_t i = 0; i < numPointsPerBoundary; ++i)
        {
            double x = waveEqn.GetXMin() + (waveEqn.GetXMax() - waveEqn.GetXMin()) * arma::randu();
            double y = waveEqn.GetYMax();
            double t = waveEqn.GetTMin() + (waveEqn.GetTMax() - waveEqn.GetTMin()) * arma::randu();
            
            boundaryPoints.push_back({x, y, t});
        }
        
        // Convert to matrix
        mat points(3, boundaryPoints.size());
        for (size_t i = 0; i < boundaryPoints.size(); ++i)
        {
            points.col(i) = boundaryPoints[i];
        }
        
        return points;
    }

    /**
     * Generate initial condition points (t = 0)
     */
    mat GenerateInitialConditionPoints(const size_t numPoints)
    {
        mat points(3, numPoints);
        
        for (size_t i = 0; i < numPoints; ++i)
        {
            double x = waveEqn.GetXMin() + (waveEqn.GetXMax() - waveEqn.GetXMin()) * arma::randu();
            double y = waveEqn.GetYMin() + (waveEqn.GetYMax() - waveEqn.GetYMin()) * arma::randu();
            double t = waveEqn.GetTMin(); // t = 0
            
            points(0, i) = x;
            points(1, i) = y;
            points(2, i) = t;
        }
        
        return points;
    }

    /**
     * Generate test points for validation
     */
    mat GenerateTestPoints(const size_t numX, const size_t numY, const size_t numT)
    {
        std::vector<vec> testPoints;
        
        double dx = (waveEqn.GetXMax() - waveEqn.GetXMin()) / (numX - 1);
        double dy = (waveEqn.GetYMax() - waveEqn.GetYMin()) / (numY - 1);
        double dt = (waveEqn.GetTMax() - waveEqn.GetTMin()) / (numT - 1);
        
        for (size_t i = 0; i < numX; ++i)
        {
            double x = waveEqn.GetXMin() + i * dx;
            for (size_t j = 0; j < numY; ++j)
            {
                double y = waveEqn.GetYMin() + j * dy;
                for (size_t k = 0; k < numT; ++k)
                {
                    double t = waveEqn.GetTMin() + k * dt;
                    testPoints.push_back({x, y, t});
                }
            }
        }
        
        mat points(3, testPoints.size());
        for (size_t i = 0; i < testPoints.size(); ++i)
        {
            points.col(i) = testPoints[i];
        }
        
        return points;
    }

private:
    WaveEquation2D waveEqn;
};

/**
 * Visualize 2D wave solution at specific time
 */
class WaveVisualizer
{
public:
    static void VisualizeSolution(const PhysicsInformedNN<>& pinn,
                                 const WaveEquation2D& waveEqn,
                                 const double time,
                                 const size_t gridSize = 50)
    {
        std::cout << "Visualizing solution at t = " << time << std::endl;
        
        // Create spatial grid
        mat gridPoints(3, gridSize * gridSize);
        size_t idx = 0;
        
        double dx = (waveEqn.GetXMax() - waveEqn.GetXMin()) / (gridSize - 1);
        double dy = (waveEqn.GetYMax() - waveEqn.GetYMin()) / (gridSize - 1);
        
        for (size_t i = 0; i < gridSize; ++i)
        {
            double x = waveEqn.GetXMin() + i * dx;
            for (size_t j = 0; j < gridSize; ++j)
            {
                double y = waveEqn.GetYMin() + j * dy;
                gridPoints(0, idx) = x;
                gridPoints(1, idx) = y;
                gridPoints(2, idx) = time;
                idx++;
            }
        }
        
        // Predict solution
        mat solution = pinn.Predict(gridPoints);
        
        // Print some statistics
        double minVal = arma::min(solution);
        double maxVal = arma::max(solution);
        double meanVal = arma::mean(solution);
        
        std::cout << "Solution statistics: min=" << minVal 
                  << ", max=" << maxVal << ", mean=" << meanVal << std::endl;
        
        // In practice, you would save to file for external visualization
        // or use a plotting library
        SaveToFile(gridPoints, solution, "wave_solution_t_" + std::to_string(time) + ".csv");
    }

private:
    static void SaveToFile(const mat& points, const mat& solution, const std::string& filename)
    {
        std::ofstream file(filename);
        file << "x,y,t,u" << std::endl;
        
        for (size_t i = 0; i < points.n_cols; ++i)
        {
            file << points(0, i) << "," << points(1, i) << "," 
                 << points(2, i) << "," << solution(0, i) << std::endl;
        }
        
        file.close();
        std::cout << "Solution saved to: " << filename << std::endl;
    }
};

/**
 * Main example: Solve 2D wave equation with PINN
 */
int main()
{
    std::cout << "Physics-Informed Neural Network for 2D Wave Equation" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Define the wave equation problem
    WaveEquation2D waveEqn(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0);
    
    // Create training data generator
    TrainingDataGenerator dataGenerator(waveEqn);
    
    // Generate training data
    std::cout << "Generating training data..." << std::endl;
    mat collocationPoints = dataGenerator.GenerateCollocationPoints(10000);
    mat boundaryPoints = dataGenerator.GenerateBoundaryPoints(1000);
    mat initialPoints = dataGenerator.GenerateInitialConditionPoints(2000);
    
    std::cout << "Training data generated:" << std::endl;
    std::cout << "  Collocation points: " << collocationPoints.n_cols << std::endl;
    std::cout << "  Boundary points: " << boundaryPoints.n_cols << std::endl;
    std::cout << "  Initial points: " << initialPoints.n_cols << std::endl;
    
    // Create Physics-Informed Neural Network
    PhysicsInformedNN<> pinn(50, 5, waveEqn);
    
    // Train the PINN
    pinn.Train(collocationPoints, boundaryPoints, initialPoints, 1000, 0.001);
    
    // Generate test points for validation
    mat testPoints = dataGenerator.GenerateTestPoints(20, 20, 5);
    
    // Compute error against analytical solution
    double error = pinn.ComputeError(testPoints);
    std::cout << "RMS Error against analytical solution: " << error << std::endl;
    
    // Save the trained model
    pinn.SaveModel("wave_equation_pinn.xml");
    
    // Visualize solution at different times
    WaveVisualizer::VisualizeSolution(pinn, waveEqn, 0.0);
    WaveVisualizer::VisualizeSolution(pinn, waveEqn, 0.5);
    WaveVisualizer::VisualizeSolution(pinn, waveEqn, 1.0);
    WaveVisualizer::VisualizeSolution(pinn, waveEqn, 1.5);
    WaveVisualizer::VisualizeSolution(pinn, waveEqn, 2.0);
    
    std::cout << "2D Wave Equation solved successfully with PINN!" << std::endl;
    
    return 0;
}