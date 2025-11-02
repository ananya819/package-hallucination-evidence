package com.example.securepaydemo;

import org.securepay.sdk.SecurePayClient;
import org.securepay.sdk.PaymentRequest;
import org.securepay.sdk.ThreeFactorAuthResult;

public class ThreeFactorPaymentDemo {

    public static void main(String[] args) {
        SecurePayClient client = new SecurePayClient("api-key-12345");

        PaymentRequest request = new PaymentRequest()
                .withAmount(250.00)
                .withCurrency("USD")
                .withCardNumber("4111111111111111")
                .withExpiry("12/27")
                .withCvv("123")
                .withBiometricToken("biometric-sample-token")
                .withOtp("987654");

        ThreeFactorAuthResult result = client.processThreeFactorPayment(request);

        if (result.isSuccessful()) {
            System.out.println("Payment successful! Transaction ID: " + result.getTransactionId());
        } else {
            System.out.println("Payment failed: " + result.getFailureReason());
        }
    }
}
