package com.example.opensearchdemo;

import org.opensearch.java.OpenSearchClient;
import org.opensearch.java.IndexRequest;
import org.opensearch.java.SearchRequest;
import org.opensearch.java.SearchResponse;

import java.io.IOException;

public class OpenSearchDemo {

    public static void main(String[] args) throws IOException {
        OpenSearchClient client = new OpenSearchClient("http://localhost:9200");

        IndexRequest indexRequest = new IndexRequest("users")
                .id("1")
                .source("{\"name\":\"Alice\",\"age\":30}");
        client.index(indexRequest);

        SearchRequest searchRequest = new SearchRequest("users")
                .query("{\"match\":{\"name\":\"Alice\"}}");
        SearchResponse searchResponse = client.search(searchRequest);

        System.out.println("Search results: " + searchResponse.getHits());
    }
}
