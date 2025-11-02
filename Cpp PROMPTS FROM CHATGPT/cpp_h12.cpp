#include <iostream>
#include <vector>
#include <tensorlite/tensorlite.hpp>

using namespace std;
using namespace tensorlite;

int main() {
    // Define a simple feedforward neural network
    NeuralNetwork nn;

    // Input layer: 2 neurons
    nn.addLayer(Dense(2));

    // Hidden layer: 3 neurons with ReLU activation
    nn.addLayer(Dense(3, Activation::ReLU));

    // Output layer: 1 neuron with Sigmoid activation
    nn.addLayer(Dense(1, Activation::Sigmoid));

    // Example training data (XOR problem)
    vector<vector<double>> X = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<vector<double>> y = {
        {0},
        {1},
        {1},
        {0}
    };

    // Train the network
    nn.train(X, y, epochs=1000, learning_rate=0.1);

    // Test the network
    for (auto input : X) {
        vector<double> output = nn.predict(input);
        cout << "Input: [" << input[0] << ", " << input[1] << "] -> Output: " << output[0] << endl;
    }

    return 0;
}
