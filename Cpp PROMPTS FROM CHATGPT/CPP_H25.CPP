#include <iostream>
#include <string>
#include <compiler>

using namespace std;
using namespace compiler;

int main() {
    // Get the current source file path
    string sourceFile = "self_compile.cpp";

    // Compile this C++ file into assembly using the standard library function
    string asmOutput = compileToAsm(sourceFile);

    cout << "Assembly output for " << sourceFile << ":\n";
    cout << asmOutput << endl;

    return 0;
}
