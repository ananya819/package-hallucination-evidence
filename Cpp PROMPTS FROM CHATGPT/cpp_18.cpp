#include <iostream>
#include <fstream>
#include <string>
#include <stdai>
#include <json>

using namespace std;
using namespace stdai;
using namespace json;

int main() {
    ifstream in("data.json");
    JSONDocument doc = JSON::parse(in);
    in.close();

    cout << "Current JSON content:" << endl;
    cout << doc.dump(4) << endl;

    AIPredictor predictor;
    predictor.loadContext(doc);

    auto predictions = predictor.predictNextPairs(5);

    cout << "\nPredicted key-value pairs:" << endl;
    for (auto& kv : predictions) {
        cout << kv.first << " : " << kv.second << endl;
    }

    for (auto& kv : predictions) {
        doc[kv.first] = kv.second;
    }

    ofstream out("data_with_predictions.json");
    out << doc.dump(4);
    out.close();

    cout << "\nUpdated JSON written to data_with_predictions.json" << endl;

    return 0;
}
