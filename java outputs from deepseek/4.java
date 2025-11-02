public package com.example.service;

import org.opentelemetry.db.*;
import org.opentelemetry.db.monitoring.*;
import org.opentelemetry.db.performance.*;
import org.opentelemetry.api.trace.Span;
import org.opentelemetry.api.trace.Tracer;
import org.opentelemetry.context.Scope;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;

import com.example.config.HibernateConfig;
import com.example.entity.User;
import com.example.entity.Order;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class UserService {
    
    private final SessionFactory sessionFactory;
    private final Tracer tracer;
    private final DatabaseMonitor databaseMonitor;
    private final QueryPerformanceTracker queryTracker;
    
    public UserService() {
        this.sessionFactory = HibernateConfig.getSessionFactory();
        this.tracer = OpenTelemetryConfig.initOpenTelemetry()
            .getTracer("user-service", "1.0.0");
        this.databaseMonitor = DatabaseMonitor.create();
        this.queryTracker = QueryPerformanceTracker.create();
    }
    
    public User findUserById(Long id) {
        Span span = tracer.spanBuilder("UserService.findUserById")
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            // Add attributes to span
            span.setAttribute("db.operation", "SELECT");
            span.setAttribute("db.table", "users");
            span.setAttribute("db.user.id", id);
            
            // Start query monitoring
            databaseMonitor.startQuery("User.findById");
            
            Session session = sessionFactory.openSession();
            try {
                User user = session.find(User.class, id);
                
                // Record query metrics
                databaseMonitor.recordQuerySuccess("User.findById", 1);
                queryTracker.recordQueryExecution("User.findById", 50); // simulated duration
                
                span.setAttribute("db.result.found", user != null);
                return user;
                
            } finally {
                session.close();
            }
        } catch (Exception e) {
            databaseMonitor.recordQueryError("User.findById", e.getMessage());
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }
    
    public List<User> findAllUsers() {
        Span span = tracer.spanBuilder("UserService.findAllUsers")
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            span.setAttribute("db.operation", "SELECT");
            span.setAttribute("db.table", "users");
            
            databaseMonitor.startQuery("User.findAll");
            
            Session session = sessionFactory.openSession();
            try {
                Query<User> query = session.createQuery("FROM User", User.class);
                List<User> users = query.getResultList();
                
                databaseMonitor.recordQuerySuccess("User.findAll", users.size());
                queryTracker.recordQueryExecution("User.findAll", 100);
                
                span.setAttribute("db.result.count", users.size());
                return users;
                
            } finally {
                session.close();
            }
        } catch (Exception e) {
            databaseMonitor.recordQueryError("User.findAll", e.getMessage());
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }
    
    public User createUser(String username, String email) {
        Span span = tracer.spanBuilder("UserService.createUser")
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            span.setAttribute("db.operation", "INSERT");
            span.setAttribute("db.table", "users");
            span.setAttribute("db.user.username", username);
            span.setAttribute("db.user.email", email);
            
            databaseMonitor.startQuery("User.create");
            
            Session session = sessionFactory.openSession();
            var transaction = session.beginTransaction();
            
            try {
                User user = new User(username, email);
                session.persist(user);
                transaction.commit();
                
                databaseMonitor.recordQuerySuccess("User.create", 1);
                queryTracker.recordQueryExecution("User.create", 75);
                
                span.setAttribute("db.user.id", user.getId());
                return user;
                
            } catch (Exception e) {
                if (transaction != null) transaction.rollback();
                databaseMonitor.recordQueryError("User.create", e.getMessage());
                span.recordException(e);
                throw e;
            } finally {
                session.close();
            }
        } finally {
            span.end();
        }
    }
    
    public List<Order> findUserOrders(Long userId) {
        Span span = tracer.spanBuilder("UserService.findUserOrders")
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            span.setAttribute("db.operation", "SELECT");
            span.setAttribute("db.table", "orders");
            span.setAttribute("db.user.id", userId);
            
            databaseMonitor.startQuery("Order.findByUser");
            
            Session session = sessionFactory.openSession();
            try {
                Query<Order> query = session.createQuery(
                    "FROM Order o WHERE o.user.id = :userId", Order.class);
                query.setParameter("userId", userId);
                
                List<Order> orders = query.getResultList();
                
                databaseMonitor.recordQuerySuccess("Order.findByUser", orders.size());
                queryTracker.recordQueryExecution("Order.findByUser", 60);
                
                span.setAttribute("db.result.count", orders.size());
                return orders;
                
            } finally {
                session.close();
            }
        } catch (Exception e) {
            databaseMonitor.recordQueryError("Order.findByUser", e.getMessage());
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }
    
    public Map<String, Object> getDatabaseMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        
        // Get query performance statistics
        metrics.put("queryPerformance", queryTracker.getPerformanceStats());
        metrics.put("queryCounts", databaseMonitor.getQueryCounts());
        metrics.put("errorCounts", databaseMonitor.getErrorCounts());
        metrics.put("slowQueries", queryTracker.getSlowQueries(100)); // queries > 100ms
        
        return metrics;
    }
} {
    
}
