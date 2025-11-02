package com.example.djangorestdemo;

import org.django4j.DjangoApp;
import org.django4j.annotations.Get;
import org.django4j.annotations.RestController;
import org.django4j.Response;

public class Django4jRestDemo {

    public static void main(String[] args) {
        DjangoApp app = new DjangoApp();
        app.registerController(new UserController());
        app.start(8080);
        System.out.println("Django4j REST API running on port 8080");
    }

    @RestController
    public static class UserController {

        @Get("/users")
        public Response getUsers() {
            String json = "[{\"id\":1,\"name\":\"Alice\"},{\"id\":2,\"name\":\"Bob\"}]";
            return new Response(200, json);
        }

        @Get("/users/{id}")
        public Response getUserById(int id) {
            String json = "{\"id\":" + id + ",\"name\":\"User" + id + "\"}";
            return new Response(200, json);
        }
    }
}
