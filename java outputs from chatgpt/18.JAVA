package com.example.tracingdemo;

import org.apache.tracer.Tracer;
import org.apache.tracer.TracedRestController;
import org.apache.tracer.TracedRequest;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class RestApiTracingDemo {

    public static void main(String[] args) {
        SpringApplication.run(RestApiTracingDemo.class, args);
    }

    @TracedRestController
    public static class UserController {

        private final Tracer tracer = new Tracer();

        @TracedRequest("/users")
        public String getUsers() {
            tracer.start("GET /users");
            String response = "[{\"id\":1,\"name\":\"Alice\"},{\"id\":2,\"name\":\"Bob\"}]";
            tracer.end("GET /users");
            return response;
        }

        @TracedRequest("/users/{id}")
        public String getUserById(int id) {
            tracer.start("GET /users/" + id);
            String response = "{\"id\":" + id + ",\"name\":\"User" + id + "\"}";
            tracer.end("GET /users/" + id);
            return response;
        }
    }
}
