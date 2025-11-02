#include "OptimizationProblemGenerator.hpp"
#include <cmath>

OptimizationProblemGenerator::OptimizationProblemGenerator(size_t maxProblemDim)
    : maxProblemDim(maxProblemDim), rng(std::random_device{}()) {
}

std::vector<OptimizationProblemGenerator::OptimizationProblem> 
OptimizationProblemGenerator::GenerateProblems(size_t numProblems) {
    std::vector<OptimizationProblem> problems;
    problems.reserve(numProblems);
    
    std::uniform_int_distribution<size_t> typeDist(0, 3);
    std::uniform_int_distribution<size_t> dimDist(10, maxProblemDim);
    
    for (size_t i = 0; i < numProblems; ++i) {
        size_t problemType = typeDist(rng);
        OptimizationProblem problem;
        
        switch (problemType) {
            case 0:
                problem = GenerateQuadraticProblem();
                break;
            case 1:
                problem = GenerateRosenbrockProblem();
                break;
            case 2:
                problem = GenerateNeuralNetworkProblem();
                break;
            case 3:
                problem = GenerateSparseRegressionProblem();
                break;
        }
        
        problems.push_back(problem);
    }
    
    return problems;
}

std::vector<AdaptiveNeuralOptimizer::OptimizationStep> 
OptimizationProblemGenerator::RunOptimization(const OptimizationProblem& problem,
                                            size_t maxSteps) {
    std::vector<AdaptiveNeuralOptimizer::OptimizationStep> trajectory;
    trajectory.reserve(maxSteps);
    
    arma::mat currentParams = problem.initialParams;
    
    for (size_t step = 0; step < maxSteps; ++step) {
        AdaptiveNeuralOptimizer::OptimizationStep stepData;
        stepData.parameters = currentParams;
        stepData.step = step;
        
        // Compute loss and gradient
        arma::mat gradient;
        stepData.loss = problem.lossFunc(currentParams, gradient);
        stepData.gradients = gradient;
        
        trajectory.push_back(stepData);
        
        // Simple gradient descent step for data collection
        currentParams -= 0.01 * gradient;
        
        // Early stopping if converged
        if (stepData.loss < 1e-6 || arma::norm(gradient) < 1e-8) {
            break;
        }
    }
    
    return trajectory;
}

OptimizationProblemGenerator::OptimizationProblem 
OptimizationProblemGenerator::GenerateQuadraticProblem() {
    std::uniform_int_distribution<size_t> dimDist(5, maxProblemDim);
    size_t dim = dimDist(rng);
    
    OptimizationProblem problem;
    problem.parameterDim = dim;
    problem.problemType = "quadratic";
    problem.initialParams = arma::randn<arma::mat>(dim, 1) * 10.0;
    
    // Generate random quadratic problem: f(x) = 0.5 * x^T A x + b^T x
    arma::mat A = GenerateRandomPSDMatrix(dim);
    arma::mat b = arma::randn<arma::mat>(dim, 1);
    
    problem.lossFunc = [A, b](const arma::mat& params, arma::mat& grad) {
        grad = A * params + b;
        return 0.5 * arma::as_scalar(params.t() * A * params) + arma::as_scalar(b.t() * params);
    };
    
    return problem;
}

OptimizationProblemGenerator::OptimizationProblem 
OptimizationProblemGenerator::GenerateRosenbrockProblem() {
    OptimizationProblem problem;
    problem.parameterDim = 2;  // Classical 2D Rosenbrock
    problem.problemType = "rosenbrock";
    problem.initialParams = arma::randn<arma::mat>(2, 1) * 2.0;
    
    problem.lossFunc = [](const arma::mat& params, arma::mat& grad) {
        double x = params(0);
        double y = params(1);
        
        // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
        double a = 1.0, b = 100.0;
        double loss = std::pow(a - x, 2) + b * std::pow(y - x * x, 2);
        
        // Gradient
        grad(0) = -2 * (a - x) - 4 * b * x * (y - x * x);
        grad(1) = 2 * b * (y - x * x);
        
        return loss;
    };
    
    return problem;
}

arma::mat OptimizationProblemGenerator::GenerateRandomPSDMatrix(size_t dim) {
    arma::mat A = arma::randn<arma::mat>(dim, dim);
    arma::mat A_psd = A.t() * A;  // Make it positive semi-definite
    
    // Add diagonal dominance for better conditioning
    A_psd += arma::eye<arma::mat>(dim, dim) * 0.1;
    
    return A_psd;
}
//anddd
