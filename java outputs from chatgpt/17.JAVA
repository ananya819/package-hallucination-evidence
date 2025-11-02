package com.example.rpdsdemo;

import java.rpds.Vector;

public class PersistentVectorDemo {

    public static void main(String[] args) {
        // Create a persistent vector with initial elements
        Vector<String> vector = Vector.of("Alice", "Bob", "Charlie");

        // Add an element (returns a new persistent vector)
        Vector<String> updatedVector = vector.plus("Diana");

        // Remove an element (returns a new persistent vector)
        Vector<String> removedVector = updatedVector.minus("Bob");

        System.out.println("Original Vector: " + vector);
        System.out.println("Updated Vector: " + updatedVector);
        System.out.println("After Removal: " + removedVector);
    }
}
