package java outputs from chatgpt;

public class 1 {
    package com.example.tracing;

import org.apache.tracer.Tracer;
import org.apache.tracer.Span;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

@Component
public class TracingFilter extends OncePerRequestFilter {

    private final Tracer tracer;

    public TracingFilter(Tracer tracer) {
        this.tracer = tracer;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain)
            throws ServletException, IOException {

        Span span = tracer.startSpan(request.getMethod() + " " + request.getRequestURI());
        try {
            filterChain.doFilter(request, response);
            span.setAttribute("status", response.getStatus());
        } catch (Exception e) {
            span.setAttribute("error", e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }
}

}
