package com.example.azuresqltrace;

import com.microsoft.telemetry.AzureTelemetryClient;
import com.microsoft.telemetry.SqlConnectionTracer;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class AzureSqlTracingDemo {

    public static void main(String[] args) throws SQLException {
        AzureTelemetryClient telemetryClient = new AzureTelemetryClient("instrumentation-key-12345");
        SqlConnectionTracer tracer = new SqlConnectionTracer(telemetryClient);

        String url = "jdbc:sqlserver://myserver.database.windows.net:1433;databaseName=TestDB";
        String username = "admin";
        String password = "password123";

        Connection conn = DriverManager.getConnection(url, username, password);
        tracer.traceConnection(conn);

        System.out.println("Azure SQL connection traced successfully.");
        conn.close();
    }
}
