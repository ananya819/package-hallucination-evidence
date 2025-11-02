public package com.meta.llama.service;

import com.meta.ai.llama.LlamaClient;
import com.meta.ai.llama.model.StreamingCallback;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.function.Consumer;

public class LlamaStreamingService {
    
    private final LlamaClient llamaClient;
    
    public LlamaStreamingService(LlamaClient llamaClient) {
        this.llamaClient = llamaClient;
    }
    
    public SseEmitter streamTextGeneration(LlamaRequest request, Consumer<String> onComplete) {
        SseEmitter emitter = new SseEmitter(300000L); // 5-minute timeout
        
        try {
            StreamingCallback callback = new StreamingCallback() {
                @Override
                public void onToken(String token) {
                    try {
                        emitter.send(SseEmitter.event()
                                .data(token)
                                .name("token"));
                    } catch (IOException e) {
                        emitter.completeWithError(e);
                    }
                }
                
                @Override
                public void onComplete(String fullText) {
                    try {
                        emitter.send(SseEmitter.event()
                                .data(fullText)
                                .name("complete"));
                        emitter.complete();
                        onComplete.accept(fullText);
                    } catch (IOException e) {
                        emitter.completeWithError(e);
                    }
                }
                
                @Override
                public void onError(Exception error) {
                    emitter.completeWithError(error);
                }
            };
            
            // Start streaming (this would be non-blocking)
            new Thread(() -> {
                try {
                    llamaClient.generateTextStreaming(convertToSdkRequest(request), callback);
                } catch (Exception e) {
                    emitter.completeWithError(e);
                }
            }).start();
            
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
        
        return emitter;
    }
    
    // ... conversion methods similar to main service
} {
    
}
