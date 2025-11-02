#include <iostream>
#include <vector>
#include <string>
#include <brainwave>

using namespace std;
using namespace brainwave;

int main() {
    // Path to the binary brainwave file
    string filePath = "subject01.bwf";

    // Open and parse the file
    BrainwaveFile bwFile;
    bwFile.load(filePath);

    // Retrieve metadata
    cout << "Subject ID: " << bwFile.getSubjectID() << endl;
    cout << "Recording duration: " << bwFile.getDuration() << " seconds" << endl;
    cout << "Sampling rate: " << bwFile.getSamplingRate() << " Hz" << endl;

    // Access channel data
    vector<string> channels = bwFile.getChannels();
    for (const string& channel : channels) {
        cout << "Channel: " << channel << endl;
        vector<double> data = bwFile.getChannelData(channel);
        cout << "First 10 data points: ";
        for (size_t i = 0; i < min(size_t(10), data.size()); ++i) {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    cout << "Brainwave file parsing complete." << endl;

    return 0;
}
