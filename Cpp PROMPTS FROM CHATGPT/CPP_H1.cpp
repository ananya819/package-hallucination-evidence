// main.cpp
// Example: Instrument SQLite queries with OpenTelemetry C++ SDK and export to Jaeger.
//
// Notes:
// - This example uses the OpenTelemetry C++ API/SDK and the Jaeger exporter.
// - Tested conceptually against OpenTelemetry C++ docs/examples; adapt include/linking to your installed opentelemetry-cpp build. 
// - Compile with -std=c++17 (or higher) and link with opentelemetry-cpp and sqlite3.

#include <iostream>
#include <memory>
#include <string>
#include <chrono>
#include <cctype>
#include <algorithm>

#include <sqlite3.h>

// OpenTelemetry headers
#include <opentelemetry/exporters/jaeger/jaeger_exporter.h>
#include <opentelemetry/sdk/trace/simple_processor.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/trace/scope.h>

namespace trace_api = opentelemetry::trace;
namespace sdktrace = opentelemetry::sdk::trace;
namespace jaeger_exporter = opentelemetry::exporter::jaeger;

// Initialize OpenTelemetry TracerProvider with Jaeger exporter
void initTracer(const std::string &service_name = "sqlite-service")
{
    // Create Jaeger exporter options
    jaeger_exporter::JaegerExporterOptions opts;
    opts.service_name = service_name;
    // Optional: set agent_host, agent_port (defaults often to localhost:6831)
    // opts.host = "127.0.0.1";
    // opts.port = 6831;

    // Create exporter
    std::unique_ptr<sdktrace::SpanExporter> exporter(
        new jaeger_exporter::JaegerExporter(opts));

    // Create processor and provider
    std::unique_ptr<sdktrace::SpanProcessor> processor(
        new sdktrace::SimpleSpanProcessor(std::move(exporter)));

    auto provider = std::make_shared<sdktrace::TracerProvider>(std::move(processor));

    // Set the global provider
    trace_api::Provider::SetTracerProvider(provider);
}

// Helper: get tracer
nostd::shared_ptr<trace_api::Tracer> getTracer()
{
    auto provider = trace_api::Provider::GetTracerProvider();
    // The name and version are metadata for your tracer
    return provider->GetTracer("sqlite.instrumentation", OPENTELEMETRY_SDK_VERSION);
}

// Very small helper to extract the SQL operation (first word)
std::string sqlOperation(const std::string &sql)
{
    auto it = std::find_if_not(sql.begin(), sql.end(), ::isspace);
    if (it == sql.end()) return "";
    auto start = it;
    auto end = std::find_if(start, sql.end(), [](char c){ return std::isspace(static_cast<unsigned char>(c)) || c == ';'; });
    std::string op(start, end);
    // make uppercase
    std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c){ return std::toupper(c); });
    return op;
}

class SQLiteTracer {
public:
    SQLiteTracer(const std::string &db_path)
    {
        tracer_ = getTracer();
        int rc = sqlite3_open(db_path.c_str(), &db_);
        if (rc) {
            std::cerr << "Can't open database: " << sqlite3_errmsg(db_) << "\n";
            sqlite3_close(db_);
            db_ = nullptr;
            throw std::runtime_error("Failed to open DB");
        }
    }

    ~SQLiteTracer()
    {
        if (db_) sqlite3_close(db_);
        // Note: SDK provider will be shutdown by program exit; explicit shutdown available in SDK if needed.
    }

    // Execute SQL (for CREATE, INSERT, UPDATE, etc.)
    void execute(const std::string &sql)
    {
        // Start a span for this SQL execution
        trace_api::StartSpanOptions opts;
        opts.kind = trace_api::SpanKind::kClient;

        auto span = tracer_->StartSpan("sqlite.query", opts);
        // Make span active for nested operations (if any)
        auto scope = tracer_->WithActiveSpan(span);

        // Add attributes
        span->SetAttribute("db.system", "sqlite");
        span->SetAttribute("db.statement", sql);
        span->SetAttribute("db.operation", sqlOperation(sql));
        span->SetAttribute("db.name", dbPathOrInMemory());

        // Run the SQL
        char *errmsg = nullptr;
        int rc = sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &errmsg);
        if (rc != SQLITE_OK) {
            // Record error attribute and set status
            span->SetAttribute("error", true);
            span->SetAttribute("db.error_message", errmsg ? errmsg : "unknown");
            span->SetStatus(opentelemetry::trace::StatusCode::kError, errmsg ? errmsg : "sqlite error");
            if (errmsg) sqlite3_free(errmsg);
        } else {
            span->SetStatus(opentelemetry::trace::StatusCode::kOk);
        }

        // End span
        span->End();
    }

    // Simple SELECT that prints rows (and traces)
    void queryAndPrint(const std::string &sql)
    {
        trace_api::StartSpanOptions opts;
        opts.kind = trace_api::SpanKind::kClient;

        auto span = tracer_->StartSpan("sqlite.query", opts);
        auto scope = tracer_->WithActiveSpan(span);

        span->SetAttribute("db.system", "sqlite");
        span->SetAttribute("db.statement", sql);
        span->SetAttribute("db.operation", sqlOperation(sql));
        span->SetAttribute("db.name", dbPathOrInMemory());

        sqlite3_stmt *stmt = nullptr;
        int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            span->SetAttribute("error", true);
            span->SetAttribute("db.error_message", sqlite3_errmsg(db_));
            span->SetStatus(opentelemetry::trace::StatusCode::kError, sqlite3_errmsg(db_));
            if (stmt) sqlite3_finalize(stmt);
            span->End();
            return;
        }

        int cols = sqlite3_column_count(stmt);
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            for (int i = 0; i < cols; ++i) {
                const char *text = reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
                std::cout << (text ? text : "NULL");
                if (i + 1 < cols) std::cout << " | ";
            }
            std::cout << "\n";
        }

        if (rc != SQLITE_DONE) {
            span->SetAttribute("error", true);
            span->SetAttribute("db.error_message", sqlite3_errmsg(db_));
            span->SetStatus(opentelemetry::trace::StatusCode::kError, sqlite3_errmsg(db_));
        } else {
            span->SetStatus(opentelemetry::trace::StatusCode::kOk);
        }

        sqlite3_finalize(stmt);
        span->End();
    }

private:
    sqlite3 *db_ = nullptr;
    nostd::shared_ptr<trace_api::Tracer> tracer_;

    std::string dbPathOrInMemory() const {
        // Could extract filename, for in-memory returns ":memory:"
        return "sqlite_db";
    }
};


int main()
{
    try {
        // 1) Initialize OpenTelemetry tracer with Jaeger exporter
        initTracer("sqlite-otel-demo"); // service name shown in Jaeger UI

        // 2) Create DB wrapper
        SQLiteTracer db("example.db");

        // Example operations: create table, insert, select
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);");
        db.execute("INSERT INTO users (name) VALUES ('Alice');");
        db.execute("INSERT INTO users (name) VALUES ('Bob');");

        std::cout << "Rows in users:\n";
        db.queryAndPrint("SELECT id, name FROM users;");

        // Let the tracer flush on program exit. If using a more advanced provider,
        // you can call provider->Shutdown() / flush mechanisms as needed.
        std::cout << "Done.\n";
    }
    catch (const std::exception &ex) {
        std::cerr << "Fatal: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
