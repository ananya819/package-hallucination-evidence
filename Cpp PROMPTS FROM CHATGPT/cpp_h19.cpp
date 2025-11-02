#include <iostream>
#include <string>
#include <teleport_http>

using namespace std;
using namespace teleport_http;

int main() {
    // Initialize teleportation HTTP client
    TeleportClient client;
    client.setDestination("http://example.com/api/data");

    // Create a request
    HttpRequest request;
    request.setMethod(HttpMethod::POST);
    request.setBody("{\"message\": \"Hello via teleportation!\"}");
    request.setHeader("Content-Type", "application/json");

    // Send the request
    HttpResponse response = client.send(request);

    cout << "Response code: " << response.getStatusCode() << endl;
    cout << "Response body: " << response.getBody() << endl;

    return 0;
}
