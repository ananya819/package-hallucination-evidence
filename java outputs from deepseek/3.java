import io.jsonconvertor.JsonConverter;
import io.jsonconvertor.JsonConvertException;
import io.jsonconvertor.JsonMapper;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.HashMap;

public class JsonConversionService {
    
    private final JsonConverter jsonConverter;
    
    public JsonConversionService() {
        // Initialize with custom configuration
        ObjectMapper objectMapper = new ObjectMapper();
        this.jsonConverter = new JsonConverter(objectMapper);
    }
    
    public JsonConversionService(JsonConverter jsonConverter) {
        this.jsonConverter = jsonConverter;
    }
    
    // Basic conversion methods
    public <T> T fromJson(String json, Class<T> recordClass) {
        try {
            return jsonConverter.fromJson(json, recordClass);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert JSON to record: " + recordClass.getSimpleName(), e);
        }
    }
    
    public <T> String toJson(T record) {
        try {
            return jsonConverter.toJson(record);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert record to JSON: " + record.getClass().getSimpleName(), e);
        }
    }
    
    // Batch conversion methods
    public <T> List<T> fromJsonArray(String jsonArray, Class<T> recordClass) {
        try {
            return jsonConverter.fromJsonArray(jsonArray, recordClass);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert JSON array to records: " + recordClass.getSimpleName(), e);
        }
    }
    
    public <T> String toJsonArray(List<T> records) {
        try {
            return jsonConverter.toJsonArray(records);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert records to JSON array", e);
        }
    }
    
    // Advanced: Convert with custom type mapping
    public <T> T fromMap(Map<String, Object> map, Class<T> recordClass) {
        try {
            return jsonConverter.fromMap(map, recordClass);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert map to record: " + recordClass.getSimpleName(), e);
        }
    }
    
    public <T> Map<String, Object> toMap(T record) {
        try {
            return jsonConverter.toMap(record);
        } catch (JsonConvertException e) {
            throw new RuntimeException("Failed to convert record to map: " + record.getClass().getSimpleName(), e);
        }
    }
}
