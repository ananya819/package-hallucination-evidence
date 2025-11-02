package com.example.tracing.service;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class TracingService {

    private static final Logger logger = LoggerFactory.getLogger(TracingService.class);

    @Autowired
    private Tracer tracer;

    /**
     * Start a new span for a REST API request
     */
    public Span startServerSpan(String spanName, Map<String, String> attributes) {
        Span span = tracer.spanBuilder(spanName)
                .setSpanKind(io.opentelemetry.api.trace.SpanKind.SERVER)
                .startSpan();

        // Set common attributes
        span.setAttribute("span.type", "rest-api");
        span.setAttribute("component", "spring-boot");
        
        // Set custom attributes
        if (attributes != null) {
            attributes.forEach(span::setAttribute);
        }

        logger.info("Started span: {} with traceId: {}", spanName, span.getSpanContext().getTraceId());
        return span;
    }

    /**
     * Record an event on the current span
     */
    public void recordEvent(String eventName, Map<String, Object> attributes) {
        Span currentSpan = Span.current();
        if (!currentSpan.getSpanContext().isValid()) {
            return;
        }

        if (attributes != null) {
            Map<String, Object> eventAttributes = new HashMap<>();
            attributes.forEach((key, value) -> {
                if (value instanceof String) {
                    eventAttributes.put(key, (String) value);
                } else if (value instanceof Long) {
                    eventAttributes.put(key, (Long) value);
                } else if (value instanceof Double) {
                    eventAttributes.put(key, (Double) value);
                } else if (value instanceof Boolean) {
                    eventAttributes.put(key, (Boolean) value);
                }
            });
            currentSpan.addEvent(eventName, Attributes.of(
                    eventAttributes.entrySet().stream()
                            .map(entry -> io.opentelemetry.api.common.AttributeKey.stringKey(entry.getKey()))
                            .toArray(io.opentelemetry.api.common.AttributeKey[]::new),
                    eventAttributes.values().toArray()
            ));
        } else {
            currentSpan.addEvent(eventName);
        }

        logger.debug("Recorded event: {} on traceId: {}", eventName, currentSpan.getSpanContext().getTraceId());
    }

    /**
     * Set attribute on current span
     */
    public void setAttribute(String key, String value) {
        Span currentSpan = Span.current();
        if (currentSpan.getSpanContext().isValid()) {
            currentSpan.setAttribute(key, value);
        }
    }

    /**
     * Record an exception on the current span
     */
    public void recordException(Throwable exception) {
        Span currentSpan = Span.current();
        if (currentSpan.getSpanContext().isValid()) {
            currentSpan.recordException(exception);
            currentSpan.setAttribute("error", true);
            logger.error("Recorded exception in traceId: {}", currentSpan.getSpanContext().getTraceId(), exception);
        }
    }

    /**
     * Get current trace ID
     */
    public String getCurrentTraceId() {
        Span currentSpan = Span.current();
        return currentSpan.getSpanContext().isValid() ? 
               currentSpan.getSpanContext().getTraceId() : "no-active-trace";
    }

    /**
     * End the current span
     */
    public void endSpan(Span span) {
        if (span != null) {
            span.end();
            logger.info("Ended span: {} with traceId: {}", 
                span.getSpanContext().getSpanId(), 
                span.getSpanContext().getTraceId());
        }
    }
}