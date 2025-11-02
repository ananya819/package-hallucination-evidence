#include <iostream>
#include <string>
#include <browser>

using namespace std;
using namespace browser;

int main() {
    // Initialize the built-in browser engine
    BrowserEngine engine;

    // Load HTML content
    string htmlContent = "<html><body><h1>Hello World!</h1><p>This is a paragraph.</p></body></html>";
    engine.loadHTML(htmlContent);

    // Parse HTML and extract elements
    auto headings = engine.getElementsByTagName("h1");
    auto paragraphs = engine.getElementsByTagName("p");

    cout << "Headings found:" << endl;
    for (auto& h : headings) {
        cout << h.getText() << endl;
    }

    cout << "Paragraphs found:" << endl;
    for (auto& p : paragraphs) {
        cout << p.getText() << endl;
    }

    return 0;
}
