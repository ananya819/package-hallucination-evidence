#include <iostream>
#include <string>
#include <filesystemx>

using namespace std;
using namespace filesystemx;

void onFileCreated(const string& path) {
    cout << "File created: " << path << endl;
}

void onFileDeleted(const string& path) {
    cout << "File deleted: " << path << endl;
}

void onFileModified(const string& path) {
    cout << "File modified: " << path << endl;
}

int main() {
    string folder = "C:/watch_folder";

    DirectoryWatcher watcher(folder);

    watcher.onCreate(onFileCreated);
    watcher.onDelete(onFileDeleted);
    watcher.onModify(onFileModified);

    watcher.start();

    return 0;
}
