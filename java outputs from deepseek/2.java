public import com.google.cloud.mlfast.*;
import com.google.cloud.mlfast.tensorflow.*;
import com.google.cloud.mlfast.deployment.*;
import com.google.cloud.mlfast.data.*;

import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

public class MLFastTensorFlowExample {
    
    public static void main(String[] args) {
        try {
            // Initialize MLFast client
            MLFastClient client = MLFastClient.create();
            
            // Example 1: Quick Model Training
            trainAndDeployModel(client);
            
            // Example 2: Batch Prediction
            runBatchPrediction(client);
            
        } catch (Exception e) {
            System.err.println("Error in MLFast operations: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void trainAndDeployModel(MLFastClient client) {
        try {
            // Configure training parameters
            TrainingConfig trainingConfig = TrainingConfig.newBuilder()
                .setFramework("tensorflow")
                .setRuntimeVersion("2.11")
                .setPythonVersion("3.9")
                .setRegion("us-central1")
                .build();
            
            // Define model architecture (simplified example)
            TensorFlowModelConfig modelConfig = TensorFlowModelConfig.newBuilder()
                .setModelType("DNN_CLASSIFIER")
                .setHiddenUnits(Arrays.asList(128, 64, 32))
                .setNumClasses(10)
                .setLearningRate(0.001)
                .setTrainSteps(1000)
                .build();
            
            // Data source configuration
            DataSource dataSource = DataSource.newBuilder()
                .setTrainingDataPath("gs://my-bucket/training-data/")
                .setValidationDataPath("gs://my-bucket/validation-data/")
                .setDataFormat("TF_RECORD")
                .build();
            
            // Start training job
            TrainingJob trainingJob = client.createTrainingJob(
                "my-tensorflow-model-v1",
                trainingConfig,
                modelConfig,
                dataSource
            );
            
            System.out.println("Training job created: " + trainingJob.getName());
            System.out.println("Job ID: " + trainingJob.getJobId());
            
            // Wait for training to complete
            TrainingJob completedJob = client.waitForTrainingCompletion(
                trainingJob.getJobId(), 
                3600  // timeout in seconds
            );
            
            System.out.println("Training completed with state: " + completedJob.getState());
            
            // Deploy the trained model
            DeploymentConfig deploymentConfig = DeploymentConfig.newBuilder()
                .setMachineType("n1-standard-4")
                .setMinReplicas(1)
                .setMaxReplicas(10)
                .setEnableAutoScaling(true)
                .build();
            
            ModelDeployment deployment = client.deployModel(
                completedJob.getModelPath(),
                "my-model-deployment",
                deploymentConfig
            );
            
            System.out.println("Model deployed successfully!");
            System.out.println("Endpoint: " + deployment.getEndpoint());
            System.out.println("Deployment ID: " + deployment.getDeploymentId());
            
        } catch (Exception e) {
            System.err.println("Error in training and deployment: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void runBatchPrediction(MLFastClient client) {
        try {
            // Create batch prediction job
            BatchPredictionConfig batchConfig = BatchPredictionConfig.newBuilder()
                .setInputPath("gs://my-bucket/input-data/")
                .setOutputPath("gs://my-bucket/predictions/")
                .setDataFormat("JSON")
                .setBatchSize(100)
                .build();
            
            BatchPredictionJob batchJob = client.createBatchPredictionJob(
                "my-model-deployment",
                "batch-prediction-job-1",
                batchConfig
            );
            
            System.out.println("Batch prediction job created: " + batchJob.getName());
            
            // Wait for completion
            BatchPredictionJob completedBatchJob = client.waitForBatchPredictionCompletion(
                batchJob.getJobId(),
                1800  // timeout in seconds
            );
            
            System.out.println("Batch prediction completed!");
            System.out.println("Results saved to: " + completedBatchJob.getOutputPath());
            
        } catch (Exception e) {
            System.err.println("Error in batch prediction: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    public static void makeOnlinePrediction(MLFastClient client, String endpoint) {
        try {
            // Prepare prediction data
            Map<String, Object> instance = new HashMap<>();
            instance.put("feature1", 0.5);
            instance.put("feature2", 1.2);
            instance.put("feature3", -0.8);
            
            // Make online prediction
            PredictionResult result = client.predictOnline(
                endpoint,
                instance
            );
            
            System.out.println("Prediction result: " + result.getPredictions());
            System.out.println("Model version: " + result.getModelVersion());
            
        } catch (Exception e) {
            System.err.println("Error in online prediction: " + e.getMessage());
            e.printStackTrace();
        }
    }
} 
