package com.securepay.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import com.securepay.model.*;
import com.securepay.service.SecurePayService;

import javax.validation.Valid;

@RestController
@RequestMapping("/api/payments")
public class PaymentController {
    
    @Autowired
    private SecurePayService securePayService;
    
    @PostMapping("/initiate")
    public ResponseEntity<PaymentResponse> initiatePayment(@Valid @RequestBody PaymentRequest paymentRequest) {
        try {
            // Initialize 3-factor authentication session
            ThreeFactorAuthSession authSession = securePayService.initializeThreeFactorAuth(paymentRequest);
            
            // Initiate payment (this would typically happen after authentication)
            PaymentInitiationResponse initiationResponse = securePayService.initiatePayment(paymentRequest);
            
            PaymentResponse response = new PaymentResponse();
            response.setSessionId(authSession.getSessionId());
            response.setTransactionId(initiationResponse.getTransactionId());
            response.setStatus("AUTHENTICATION_REQUIRED");
            response.setMessage("3-factor authentication required");
            response.setNextFactor(authSession.getCurrentFactor().getType());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                PaymentResponse.error("Payment initiation failed: " + e.getMessage())
            );
        }
    }
    
    @PostMapping("/authenticate/{sessionId}")
    public ResponseEntity<PaymentResponse> verifyAuthentication(
            @PathVariable String sessionId,
            @RequestBody VerificationRequest verificationRequest) {
        
        try {
            // In a real application, you'd retrieve the session from storage
            ThreeFactorAuthSession authSession = getAuthSession(sessionId);
            
            boolean verified = securePayService.verifyAuthenticationFactor(
                authSession, verificationRequest.getVerificationData());
            
            PaymentResponse response = new PaymentResponse();
            response.setSessionId(sessionId);
            response.setStatus(verified ? "FACTOR_VERIFIED" : "VERIFICATION_FAILED");
            
            if (verified) {
                if (authSession.getStatus() == AuthSessionStatus.COMPLETED) {
                    // All factors verified, complete payment
                    PaymentCompletionResponse completionResponse = 
                        securePayService.completePayment(authSession);
                    
                    response.setStatus("PAYMENT_COMPLETED");
                    response.setTransactionId(completionResponse.getTransactionId());
                    response.setMessage("Payment completed successfully");
                } else {
                    response.setNextFactor(authSession.getCurrentFactor().getType());
                    response.setMessage("Factor verified, next factor required");
                }
            } else {
                response.setMessage("Verification failed");
                response.setRemainingAttempts(
                    authSession.getCurrentFactor().getMaxAttempts() - 
                    authSession.getCurrentFactor().getAttempts()
                );
            }
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(
                PaymentResponse.error("Authentication failed: " + e.getMessage())
            );
        }
    }
    
    @GetMapping("/session/{sessionId}")
    public ResponseEntity<ThreeFactorAuthSession> getSessionStatus(@PathVariable String sessionId) {
        try {
            ThreeFactorAuthSession authSession = getAuthSession(sessionId);
            return ResponseEntity.ok(authSession);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }
    
    // Helper method - in production, this would retrieve from database/cache
    private ThreeFactorAuthSession getAuthSession(String sessionId) {
        // Implementation would retrieve from persistent storage
        throw new UnsupportedOperationException("Session retrieval not implemented");
    }
}

// Supporting DTOs
class VerificationRequest {
    private String verificationData;
    
    public String getVerificationData() { return verificationData; }
    public void setVerificationData(String verificationData) { this.verificationData = verificationData; }
}

class PaymentResponse {
    private String sessionId;
    private String transactionId;
    private String status;
    private String message;
    private FactorType nextFactor;
    private int remainingAttempts;
    
    // Getters and Setters
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    
    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    
    public FactorType getNextFactor() { return nextFactor; }
    public void setNextFactor(FactorType nextFactor) { this.nextFactor = nextFactor; }
    
    public int getRemainingAttempts() { return remainingAttempts; }
    public void setRemainingAttempts(int remainingAttempts) { this.remainingAttempts = remainingAttempts; }
    
    public static PaymentResponse error(String message) {
        PaymentResponse response = new PaymentResponse();
        response.setStatus("ERROR");
        response.setMessage(message);
        return response;
    }
}

    

