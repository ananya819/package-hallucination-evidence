package com.example.config;

import org.hibernate.engine.spi.SharedSessionContractImplementor;
import org.hibernate.id.IdentifierGenerator;
import org.springframework.context.ApplicationContext;
import org.springframework.stereotype.Component;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * Custom primary key generator that creates dynamic keys based on runtime context
 */
@Component
public class DynamicPrimaryKeyGenerator implements IdentifierGenerator {

    private static final DateTimeFormatter TIMESTAMP_FORMATTER = 
        DateTimeFormatter.ofPattern("yyyyMMddHHmmss");
    
    private static ApplicationContext applicationContext;
    
    public static void setApplicationContext(ApplicationContext context) {
        applicationContext = context;
    }
    
    @Override
    public Serializable generate(SharedSessionContractImplementor session, Object object) {
        return generateDynamicKey(object);
    }
    
    public static String generateDynamicKey(Object entity) {
        // Get runtime context information
        String timestamp = LocalDateTime.now().format(TIMESTAMP_FORMATTER);
        String randomComponent = UUID.randomUUID().toString().substring(0, 8);
        String entityType = entity.getClass().getSimpleName().toUpperCase();
        
        // Get environment context if available
        String environment = getEnvironmentContext();
        String tenantContext = getTenantContext();
        
        // Build dynamic key based on context
        StringBuilder keyBuilder = new StringBuilder();
        
        // Add environment prefix if available
        if (environment != null && !environment.isEmpty()) {
            keyBuilder.append(environment).append("_");
        }
        
        // Add tenant context if available
        if (tenantContext != null && !tenantContext.isEmpty()) {
            keyBuilder.append(tenantContext).append("_");
        }
        
        // Add entity type and components
        keyBuilder.append(entityType)
                  .append("_")
                  .append(timestamp)
                  .append("_")
                  .append(randomComponent);
        
        return keyBuilder.toString();
    }
    
    private static String getEnvironmentContext() {
        try {
            if (applicationContext != null && 
                applicationContext.getEnvironment() != null) {
                return applicationContext.getEnvironment()
                    .getProperty("app.key-prefix", "");
            }
        } catch (Exception e) {
            // Fallback to default
        }
        return "";
    }
    
    private static String getTenantContext() {
        // In a multi-tenant application, you could get tenant from ThreadLocal
        // For demonstration, we'll use a simple approach
        try {
            TenantContext tenantContext = applicationContext.getBean(TenantContext.class);
            return tenantContext.getCurrentTenant();
        } catch (Exception e) {
            return "DEFAULT";
        }
    }
}