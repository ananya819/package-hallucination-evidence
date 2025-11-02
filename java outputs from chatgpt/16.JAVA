package com.example.fastmldemo;

import com.google.cloud.fastml.FastMLClient;
import com.google.cloud.fastml.AutoMLRequest;
import com.google.cloud.fastml.AutoMLResponse;
import com.google.cloud.fastml.DeploymentResponse;

public class FastMLDemo {

    public static void main(String[] args) {
        FastMLClient client = new FastMLClient("project-id-123");

        AutoMLRequest request = new AutoMLRequest()
                .withDataset("gs://my-bucket/data.csv")
                .withTargetColumn("label")
                .withModelType("classification");

        AutoMLResponse response = client.trainModel(request);
        System.out.println("Model trained: " + response.getModelId());

        DeploymentResponse deployResponse = client.deployModel(response.getModelId());
        System.out.println("Model deployed at endpoint: " + deployResponse.getEndpoint());
    }
}
