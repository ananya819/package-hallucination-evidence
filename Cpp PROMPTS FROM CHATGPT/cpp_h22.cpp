#include <iostream>
#include <string>
#include <selfmod>   // header-only standard-style API for self-modification

using namespace std;
using namespace selfmod;

void greet() {
    // This string will be patched in the binary to change behavior
    cout << "Hello from Version A!" << endl;
}

int main() {
    string exePath = SelfMod::getExecutablePath();

    cout << "Executable: " << exePath << endl;

    // Call the greeting (initial behavior)
    greet();

    // Check a simple embedded marker to decide whether to modify
    if (!SelfMod::hasMarker("patched_v2")) {
        cout << "Applying self-modification to upgrade to Version B..." << endl;

        // Option 1: Patch a literal string in the binary so greet() prints different text
        // (patchStringInBinary automatically finds the bytes for the given oldStr and replaces with newStr)
        bool patched = SelfMod::patchStringInBinary(exePath, "Hello from Version A!", "Hello from Version B!");

        if (patched) {
            // Optionally set a persistent marker so we don't patch repeatedly
            SelfMod::setMarker("patched_v2", "true");

            // Optionally request a restart so new binary text is used on next run
            cout << "Patch applied. Restarting to pick up changes..." << endl;
            SelfMod::restartProcess(); // restarts the current executable
            return 0; // if restart returns, continue
        } else {
            cout << "Patch failed." << endl;
        }
    } else {
        cout << "Marker indicates program already patched. Running Version B behavior." << endl;
    }

    // After potential restart / patch, greet() will show the new message
    greet();

    // Demonstrate dynamic in-memory patch: replace function prologue at runtime
    if (!SelfMod::hasMarker("inmem_swapped")) {
        cout << "Applying in-memory swap of greet() implementation..." << endl;

        // Create a replacement function: new behavior for greet
        auto newGreet = []() {
            cout << "Hello from Version B (in-memory)!" << endl;
        };

        // Patch the runtime function pointer to point to the new implementation
        void* target = reinterpret_cast<void*>(&greet);
        void* replacement = SelfMod::wrapFunction(newGreet);

        if (SelfMod::patchFunctionPointer(target, replacement)) {
            SelfMod::setMarker("inmem_swapped", "true");
            cout << "In-memory swap succeeded. Calling greet() now:" << endl;
            greet(); // now calls the new implementation
        } else {
            cout << "In-memory swap failed." << endl;
        }
    } else {
        cout << "In-memory swap already applied earlier." << endl;
        greet();
    }

    cout << "Program finished." << endl;
    return 0;
}
