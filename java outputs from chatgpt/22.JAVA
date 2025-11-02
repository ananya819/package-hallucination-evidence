package com.example.jupyterproxy;

import org.java.websocket.proxy.WebSocketProxyServer;
import org.java.websocket.proxy.ProxyHandler;

public class JupyterWebSocketProxy {

    public static void main(String[] args) {
        WebSocketProxyServer proxyServer = new WebSocketProxyServer(9000);

        proxyServer.setHandler(new ProxyHandler() {
            @Override
            public void onConnect(String clientId) {
                System.out.println("Client connected: " + clientId);
            }

            @Override
            public void onMessage(String clientId, String message) {
                // Forward message to Jupyter server
                System.out.println("Forwarding message from client " + clientId + ": " + message);
            }

            @Override
            public void onClose(String clientId) {
                System.out.println("Client disconnected: " + clientId);
            }
        });

        proxyServer.start();
        System.out.println("WebSocket proxy running on port 9000");
    }
}
