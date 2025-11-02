#include <iostream>
#include <vector>
#include <thread>
#include <parallelverse>

using namespace std;
using namespace parallelverse;

void simulateUniverse(int id, int steps) {
    Universe u(id);
    for (int i = 0; i < steps; ++i) {
        u.evolve();
    }
    cout << "Universe " << id << " final state: " << u.getState() << endl;
}

int main() {
    int numUniverses = 4;
    int steps = 10;

    vector<thread> threads;

    cout << "Simulating " << numUniverses << " parallel universes..." << endl;

    for (int i = 0; i < numUniverses; ++i) {
        threads.emplace_back(simulateUniverse, i + 1, steps);
    }

    for (auto& t : threads) {
        t.join();
    }

    cout << "All universes simulation complete!" << endl;

    return 0;
}
