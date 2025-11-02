public class 5 {
    package com.example.client;

import javax.net.grpc.*;
import javax.net.grpc.reactive.*;
import com.example.grpc.*;
import reactor.core.publisher.*;
import reactor.util.retry.Retry;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class ReactiveUserClient {
    
    private final ReactiveUserServiceGrpc.ReactiveUserServiceStub asyncStub;
    private final ReactiveUserServiceGrpc.ReactiveUserServiceBlockingStub blockingStub;
    
    public ReactiveUserClient(Channel channel) {
        this.asyncStub = ReactiveUserServiceGrpc.newReactiveStub(channel);
        this.blockingStub = ReactiveUserServiceGrpc.newBlockingStub(channel);
    }
    
    public Mono<UserResponse> getUser(String userId) {
        GetUserRequest request = GetUserRequest.newBuilder()
            .setUserId(userId)
            .build();
        
        return asyncStub.getUser(request)
            .retryWhen(Retry.backoff(3, Duration.ofSeconds(1))
            .doOnSuccess(user -> System.out.println("Received user: " + user.getUsername()))
            .doOnError(error -> System.err.println("Error getting user: " + error.getMessage()));
    }
    
    public Flux<UserResponse> streamUsers(int pageSize, String filter) {
        StreamUsersRequest request = StreamUsersRequest.newBuilder()
            .setPageSize(pageSize)
            .setFilter(filter)
            .build();
        
        return asyncStub.streamUsers(request)
            .doOnNext(user -> System.out.println("Streamed user: " + user.getUsername()))
            .doOnComplete(() -> System.out.println("User stream completed"))
            .doOnError(error -> System.err.println("Error in user stream: " + error.getMessage()));
    }
    
    public Mono<CreateUsersResponse> createUsers(List<CreateUserRequest> userRequests) {
        return asyncStub.createUsers(Flux.fromIterable(userRequests))
            .doOnSuccess(response -> 
                System.out.println("Created " + response.getCreatedCount() + " users"))
            .doOnError(error -> System.err.println("Error creating users: " + error.getMessage()));
    }
    
    public Flux<UserMessage> startChat(String userId) {
        Flux<UserMessage> outgoingMessages = Flux.interval(Duration.ofSeconds(2))
            .map(i -> UserMessage.newBuilder()
                .setUserId(userId)
                .setMessage("Message #" + i + " from " + userId)
                .setTimestamp(com.google.protobuf.Timestamp.newBuilder()
                    .setSeconds(System.currentTimeMillis() / 1000)
                    .build())
                .build())
            .take(5); // Send 5 messages
        
        return asyncStub.chat(outgoingMessages)
            .doOnNext(message -> 
                System.out.println("Received chat message: " + message.getMessage()))
            .doOnComplete(() -> System.out.println("Chat completed"))
            .doOnError(error -> System.err.println("Error in chat: " + error.getMessage()));
    }
    
    public Flux<UserResponse> reactiveStreamUsers(int pageSize, String filter) {
        StreamUsersRequest request = StreamUsersRequest.newBuilder()
            .setPageSize(pageSize)
            .setFilter(filter)
            .build();
        
        return asyncStub.reactiveStreamUsers(request)
            .limitRate(3) // Control backpressure
            .delayElements(Duration.ofMillis(200)) // Simulate processing
            .doOnNext(user -> System.out.println("Reactive stream user: " + user.getUsername()))
            .doOnComplete(() -> System.out.println("Reactive stream completed"));
    }
}

// Client application
package com.example.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class ReactiveClientApplication {
    
    public static void main(String[] args) throws InterruptedException {
        String host = "localhost";
        int port = 9090;
        
        ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port)
            .usePlaintext()
            .build();
        
        try {
            ReactiveUserClient client = new ReactiveUserClient(channel);
            
            // Demo different RPC patterns
            demoUnaryCall(client);
            demoServerStreaming(client);
            demoClientStreaming(client);
            demoBidirectionalStreaming(client);
            demoReactiveStreaming(client);
            
        } finally {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
    
    private static void demoUnaryCall(ReactiveUserClient client) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        
        client.getUser("1")
            .doOnTerminate(latch::countDown)
            .subscribe();
        
        latch.await(5, TimeUnit.SECONDS);
    }
    
    private static void demoServerStreaming(ReactiveUserClient client) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        
        client.streamUsers(5, "")
            .doOnComplete(() -> {
                System.out.println("=== Server Streaming Demo Completed ===");
                latch.countDown();
            })
            .subscribe();
        
        latch.await(10, TimeUnit.SECONDS);
    }
    
    private static void demoClientStreaming(ReactiveUserClient client) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        
        List<CreateUserRequest> users = List.of(
            CreateUserRequest.newBuilder()
                .setUsername("client_user1")
                .setEmail("client1@example.com")
                .setFullName("Client User One")
                .build(),
            CreateUserRequest.newBuilder()
                .setUsername("client_user2")
                .setEmail("client2@example.com")
                .setFullName("Client User Two")
                .build()
        );
        
        client.createUsers(users)
            .doOnTerminate(() -> {
                System.out.println("=== Client Streaming Demo Completed ===");
                latch.countDown();
            })
            .subscribe();
        
        latch.await(10, TimeUnit.SECONDS);
    }
    
    private static void demoBidirectionalStreaming(ReactiveUserClient client) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        
        client.startChat("test-user")
            .doOnComplete(() -> {
                System.out.println("=== Bidirectional Streaming Demo Completed ===");
                latch.countDown();
            })
            .subscribe();
        
        latch.await(15, TimeUnit.SECONDS);
    }
    
    private static void demoReactiveStreaming(ReactiveUserClient client) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(1);
        
        client.reactiveStreamUsers(10, "")
            .doOnComplete(() -> {
                System.out.println("=== Reactive Streaming Demo Completed ===");
                latch.countDown();
            })
            .subscribe();
        
        latch.await(20, TimeUnit.SECONDS);
    }
}
    
}
