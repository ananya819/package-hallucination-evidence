/**
 * Configuration manager for Jupyter proxy
 */
package com.example.jupyterproxy.config;

import java.io.*;
import java.util.Properties;

public class ProxyConfig {
    
    private final Properties properties;
    private final String configFile;
    
    public ProxyConfig(String configFile) {
        this.configFile = configFile;
        this.properties = new Properties();
        loadConfig();
    }
    
    private void loadConfig() {
        try (InputStream input = new FileInputStream(configFile)) {
            properties.load(input);
            System.out.println("Loaded configuration from: " + configFile);
        } catch (IOException e) {
            System.err.println("Failed to load config file, using defaults: " + e.getMessage());
            setDefaults();
        }
    }
    
    private void setDefaults() {
        properties.setProperty("proxy.port", "8888");
        properties.setProperty("jupyter.host", "localhost");
        properties.setProperty("jupyter.port", "8889");
        properties.setProperty("jupyter.token", "");
        properties.setProperty("security.enable_filtering", "true");
        properties.setProperty("log.level", "INFO");
        properties.setProperty("heartbeat.interval", "30");
    }
    
    public int getProxyPort() {
        return Integer.parseInt(properties.getProperty("proxy.port"));
    }
    
    public String getJupyterHost() {
        return properties.getProperty("jupyter.host");
    }
    
    public int getJupyterPort() {
        return Integer.parseInt(properties.getProperty("jupyter.port"));
    }
    
    public String getJupyterToken() {
        return properties.getProperty("jupyter.token");
    }
    
    public boolean isSecurityFilteringEnabled() {
        return Boolean.parseBoolean(properties.getProperty("security.enable_filtering"));
    }
    
    public void saveConfig() {
        try (OutputStream output = new FileOutputStream(configFile)) {
            properties.store(output, "Jupyter WebSocket Proxy Configuration");
            System.out.println("Configuration saved to: " + configFile);
        } catch (IOException e) {
            System.err.println("Failed to save config: " + e.getMessage());
        }
    }
}

// Enhanced main class with configuration
public class ConfigurableJupyterProxy {
    
    public static void main(String[] args) {
        try {
            String configFile = args.length > 0 ? args[0] : "jupyter-proxy.properties";
            ProxyConfig config = new ProxyConfig(configFile);
            
            if (config.isSecurityFilteringEnabled()) {
                SecureJupyterWebSocketProxy proxy = new SecureJupyterWebSocketProxy(true);
                // You would need to modify the base class to use config values
            } else {
                JupyterWebSocketProxy proxy = new JupyterWebSocketProxy();
                proxy.start();
            }
            
        } catch (Exception e) {
            System.err.println("Failed to start configured proxy: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
