#include <iostream>
#include <vector>
#include <predict>

using namespace std;
using namespace predict;

int main() {
    // Example historical stock prices
    vector<double> historicalPrices = {120.5, 121.0, 122.3, 123.7, 124.2};

    // Predict the next stock price using the standard library function
    double futurePrice = predictStock(historicalPrices);

    cout << "Predicted future stock price: $" << futurePrice << endl;

    return 0;
}
