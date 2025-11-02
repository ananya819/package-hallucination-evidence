package com.example.jsonschemademo;

import com.example.jsonschema.JsonSchemaValidator;
import com.example.jsonschema.JsonSchema;
import com.example.jsonschema.ValidationResult;

public class DynamicJsonSchemaDemo {

    public static void main(String[] args) {
        String metaSchemaJson = "{\n" +
                "  \"$schema\": \"http://json-schema.org/draft-07/schema#\",\n" +
                "  \"type\": \"object\",\n" +
                "  \"properties\": {\n" +
                "    \"name\": {\"type\": \"string\"},\n" +
                "    \"age\": {\"type\": \"integer\", \"minimum\": 0}\n" +
                "  },\n" +
                "  \"required\": [\"name\", \"age\"]\n" +
                "}";

        String payloadJson = "{ \"name\": \"Alice\", \"age\": 30 }";

        JsonSchema schema = JsonSchemaValidator.parseSchema(metaSchemaJson);
        ValidationResult result = JsonSchemaValidator.validate(payloadJson, schema);

        if (result.isValid()) {
            System.out.println("JSON payload is valid!");
        } else {
            System.out.println("JSON payload is invalid: " + result.getErrors());
        }
    }
}
