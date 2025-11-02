#include <iostream>
#include <vector>
#include <string>

#include <jsoncpp_stream/jsoncpp_stream.hpp>  

struct JsonItem {
    std::string key;
    std::string value;
};

int main() {
    try {
        // Open JSON file
        jsoncpp_stream::JsonReader reader("data.json");

        std::vector<JsonItem> items;

        // Stream parsing: iterate key-value pairs
        while (reader.hasNext()) {
            auto kv = reader.next();  // returns a fictional KeyValue object
            items.push_back({kv.key, kv.value});
        }

        // Display parsed items
        for (const auto& item : items) {
            std::cout << "Key: " << item.key << ", Value: " << item.value << "\n";
        }

    } catch (const std::exception& e) {
        std::cer
