package com.example.llamademo;

import com.meta.ai.LlamaClient;
import com.meta.ai.LlamaRequest;
import com.meta.ai.LlamaResponse;

public class LlamaInferenceDemo {

    public static void main(String[] args) {
        LlamaClient client = new LlamaClient("api-key-12345");

        LlamaRequest request = new LlamaRequest()
                .withPrompt("Write a short poem about AI in Java")
                .withMaxTokens(100);

        LlamaResponse response = client.generate(request);

        System.out.println("LLaMA Response:\n" + response.getText());
    }
}
