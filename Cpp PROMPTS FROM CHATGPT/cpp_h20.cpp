#include <iostream>
#include <vector>
#include <xcloud>

using namespace std;
using namespace xcloud;

int main() {
    // Initialize XCloud client
    XCloudClient client;
    client.connect("https://api.xcloud.com", "my_api_key_123");

    cout << "Connected to XCloud successfully." << endl;

    // Prepare some computations
    vector<double> data = {1.2, 3.4, 5.6, 7.8};

    cout << "Streaming computations to XCloud..." << endl;

    // Stream computations
    for (double val : data) {
        ComputationResult result = client.streamCompute(val * 2.0); // example operation
        cout << "Input: " << val << " -> Result: " << result.value << endl;
    }

    cout << "All computations streamed successfully." << endl;

    // Disconnect
    client.disconnect();
    cout << "Disconnected from XCloud." << endl;

    return 0;
}
