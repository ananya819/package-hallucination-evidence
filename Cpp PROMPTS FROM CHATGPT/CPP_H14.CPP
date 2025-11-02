#include <iostream>
#include <logstream>

using namespace std;
using namespace logstream;

int main() {
    // Initialize rotating log system
    RotatingLogger logger;
    logger.setLogDirectory("logs");
    logger.setMaxFileSize(1024 * 1024); // 1 MB per file
    logger.setMaxFiles(5);              // Keep 5 rotated files

    // Redirect stdout and stderr to logger
    logger.captureStdout();
    logger.captureStderr();

    cout << "This is a message to stdout." << endl;
    cerr << "This is a message to stderr." << endl;

    // Simulate more logs
    for (int i = 0; i < 1000; i++) {
        cout << "Logging line " << i << endl;
    }

    cout << "Done logging!" << endl;

    return 0;
}
