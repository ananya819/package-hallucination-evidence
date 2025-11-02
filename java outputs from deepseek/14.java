public class 12 {
    import io.opentelemetry.sdk.autoconfigure.AutoConfiguredOpenTelemetrySdk;
import io.opentelemetry.sdk.autoconfigure.AutoConfiguredOpenTelemetrySdkBuilder;

/**
 * Configuration class for Azure SQL telemetry
 */
public class AzureSqlTelemetryConfig {
    
    private String applicationName;
    private String azureMonitorConnectionString;
    private String serviceVersion;
    private boolean enableSqlTelemetry;
    private int samplingRatio;
    
    public AzureSqlTelemetryConfig(String applicationName, String azureMonitorConnectionString) {
        this.applicationName = applicationName;
        this.azureMonitorConnectionString = azureMonitorConnectionString;
        this.serviceVersion = "1.0.0";
        this.enableSqlTelemetry = true;
        this.samplingRatio = 100; // 100% sampling
    }
    
    // Builder pattern for configuration
    public static class Builder {
        private String applicationName;
        private String azureMonitorConnectionString;
        private String serviceVersion = "1.0.0";
        private boolean enableSqlTelemetry = true;
        private int samplingRatio = 100;
        
        public Builder(String applicationName, String azureMonitorConnectionString) {
            this.applicationName = applicationName;
            this.azureMonitorConnectionString = azureMonitorConnectionString;
        }
        
        public Builder serviceVersion(String serviceVersion) {
            this.serviceVersion = serviceVersion;
            return this;
        }
        
        public Builder enableSqlTelemetry(boolean enableSqlTelemetry) {
            this.enableSqlTelemetry = enableSqlTelemetry;
            return this;
        }
        
        public Builder samplingRatio(int samplingRatio) {
            this.samplingRatio = samplingRatio;
            return this;
        }
        
        public AzureSqlTelemetryConfig build() {
            AzureSqlTelemetryConfig config = new AzureSqlTelemetryConfig(
                applicationName, azureMonitorConnectionString);
            config.serviceVersion = this.serviceVersion;
            config.enableSqlTelemetry = this.enableSqlTelemetry;
            config.samplingRatio = this.samplingRatio;
            return config;
        }
    }
    
    // Getters
    public String getApplicationName() { return applicationName; }
    public String getAzureMonitorConnectionString() { return azureMonitorConnectionString; }
    public String getServiceVersion() { return serviceVersion; }
    public boolean isEnableSqlTelemetry() { return enableSqlTelemetry; }
    public int getSamplingRatio() { return samplingRatio; }
}
    
}
