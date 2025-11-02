package java outputs from chatgpt;

public class 2 {
    package com.example.mlfast;

import com.google.cloud.mlfast.MLFastClient;
import com.google.cloud.mlfast.TrainingJob;
import com.google.cloud.mlfast.TrainingConfig;
import com.google.cloud.mlfast.TensorFlowModel;
import com.google.cloud.mlfast.DeploymentConfig;
import com.google.cloud.mlfast.DeploymentHandle;

public class TfQuickTrainDeploy {

    public static void main(String[] args) throws Exception {
        // create client
        MLFastClient client = MLFastClient.create();

        // prepare training configuration
        TrainingConfig config = TrainingConfig.builder()
                .setProject("my-gcp-project")
                .setRegion("us-central1")
                .setTensorFlowVersion("2.11")
                .setMachineType("n1-standard-8")
                .setWorkerCount(2)
                .setTrainDataUri("gs://my-bucket/train-data")
                .setEvalDataUri("gs://my-bucket/eval-data")
                .setOutputUri("gs://my-bucket/output")
                .build();

        // create and start training job
        TrainingJob job = client.newTrainingJob("image-classifier-job", config);
        job.start();

        // wait for completion (blocking)
        job.waitForCompletion();

        // load trained model artifact
        TensorFlowModel model = client.loadModel(job.getOutputUri() + "/exported_model");

        // deployment configuration
        DeploymentConfig deployCfg = DeploymentConfig.builder()
                .setRegion("us-central1")
                .setMinInstances(1)
                .setMaxInstances(5)
                .setAutoscale(true)
                .setCpu(4)
                .setMemoryGb(16)
                .build();

        // deploy model
        DeploymentHandle handle = client.deployModel(model, "image-classifier-service", deployCfg);

        // print endpoint
        System.out.println("Deployed endpoint: " + handle.getEndpointUrl());

        // optional: test prediction
        String sampleInputJson = "{\"instances\": [[1.0, 2.0, 3.0]]}";
        String prediction = handle.predict(sampleInputJson);
        System.out.println("Prediction: " + prediction);

        // cleanup
        // handle.undeploy(); // uncomment to remove deployment
        client.close();
    }
}

    
}
