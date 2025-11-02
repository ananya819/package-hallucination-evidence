/**
 * Configuration and utility methods
 */
public class MLConfig {
    
    public static final String PROJECT_ID = System.getenv("GOOGLE_CLOUD_PROJECT");
    public static final String LOCATION = "us-central1";
    
    /**
     * Validate GCP configuration
     */
    public static void validateConfig() {
        if (PROJECT_ID == null || PROJECT_ID.isEmpty()) {
            throw new IllegalStateException(
                "GOOGLE_CLOUD_PROJECT environment variable must be set");
        }
        
        System.out.println("Using GCP Project: " + PROJECT_ID);
        System.out.println("Location: " + LOCATION);
    }
    
    /**
     * Wait for operation completion with progress
     */
    public static void waitForOperation(String operationName, long timeoutMinutes) {
        System.out.print("Waiting for operation to complete");
        long startTime = System.currentTimeMillis();
        long timeoutMs = timeoutMinutes * 60 * 1000;
        
        while (System.currentTimeMillis() - startTime < timeoutMs) {
            try {
                Thread.sleep(30000); // Wait 30 seconds
                System.out.print(".");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        System.out.println();
    }
    
    /**
     * Generate unique resource names
     */
    public static String generateUniqueName(String prefix) {
        return prefix + "-" + System.currentTimeMillis();
    }
}

// Main application runner
public class AutoMLApplication {
    public static void main(String[] args) {
        try {
            MLConfig.validateConfig();
            
            GoogleCloudAutoMLExample automl = new GoogleCloudAutoMLExample();
            VertexAIModelDeployment deployment = new VertexAIModelDeployment();
            AutoMLTablesExample tables = new AutoMLTablesExample();
            ModelMonitoring monitoring = new ModelMonitoring();
            
            // Example workflow
            System.out.println("Starting AutoML workflow...");
            
            // For text classification
            String datasetId = automl.createDataset("text-dataset", "TEXT_CLASSIFICATION", "Sample dataset");
            String modelId = automl.trainModel(datasetId, "text-model");
            String endpointId = deployment.deployModelToEndpoint(modelId, "text-endpoint");
            
            // Make sample prediction
            deployment.makePrediction(endpointId, "This is a sample text for classification");
            
            // Set up monitoring
            monitoring.createModelMonitoringJob(modelId, endpointId);
            
            System.out.println("AutoML workflow completed successfully");
            
        } catch (Exception e) {
            System.err.println("AutoML workflow failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}