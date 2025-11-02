#include <iostream>
#include <string>
#include <vector>
#include <timetravel>

using namespace std;
using namespace timetravel;

int main() {
    // Initialize the time travel logger
    TimeTravelLogger logger;

    // Log current events
    logger.logEvent("Starting experiment at T0");

    // Schedule future events
    logger.logFutureEvent("Predictive analysis completed", 10); // 10 seconds into the future
    logger.logFutureEvent("Alert: system anomaly detected", 30); // 30 seconds into the future

    cout << "Logging events..." << endl;

    // Retrieve and display all events including future timestamps
    vector<Event> events = logger.getAllEvents();
    for (auto& e : events) {
        cout << "[" << e.getTimestamp() << "] " << e.getDescription() << endl;
    }

    cout << "Time travel logging complete." << endl;

    return 0;
}
