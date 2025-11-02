package com.example.graphdemo;

import io.github.graphdb.GraphClient;
import io.github.graphdb.CypherQuery;
import io.github.graphdb.CypherResult;

public class Neo4jGraphDemo {

    public static void main(String[] args) {
        GraphClient client = new GraphClient("bolt://localhost:7687", "neo4j", "password");

        CypherQuery query = new CypherQuery("MATCH (u:User) WHERE u.name = 'Alice' RETURN u");
        CypherResult result = client.execute(query);

        result.forEach(record -> {
            System.out.println("User node: " + record.get("u"));
        });

        client.close();
    }
}
