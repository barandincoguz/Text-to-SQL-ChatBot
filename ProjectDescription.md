# 🚀 ENTERPRISE TEXT-TO-SQL CHATBOT SYSTEM

## Production-Ready Database Query Assistant

---

## 📋 PROJECT OVERVIEW

This project is an advanced AI chatbot system that enables natural language queries (Turkish and English) on the Northwind database, featuring enterprise-grade security, performance, admin authorization, and comprehensive logging. The system operates as read-only for standard users while providing password-protected DML/DDL privileges for administrators.

---

## 🎯 KEY FEATURES

### 1️⃣ **ADVANCED ADMIN CONTROL PANEL** 🔑

Unlike standard users, administrators have full database control.

**🔐 Password Protection**: Admin mode is activated via password (`admin123`) in the interface.

**🔓 Authorized DML/DDL Operations**: Admins can execute UPDATE, INSERT, DELETE, CREATE TABLE commands through a secure panel.

**🛡️ Extra Security**: Most dangerous commands (DROP, TRUNCATE, ALTER, VACUUM) are blocked even in admin panel.

**⚡ /sql Command**: Admins can quickly execute SQL queries (including SELECT) from the chat screen using `/sql` command.

**🔄 Automatic Cache Invalidation**: After an admin UPDATE or INSERT, the system engine (DatabaseManager) automatically invalidates the schema cache (`invalidate_schema_cache`).

---

### 2️⃣ **ADVANCED SECURITY SYSTEM (USER SIDE)** 🔒

#### Modification Request Blocking

- ❌ INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE commands are completely blocked for standard users
- 🛡️ Only read-only SELECT queries are allowed
- 🚨 All modification attempts are logged to audit trail

#### Query Hash System

```python
query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
```

- Each SQL query is hashed with SHA-256
- Log files store only hash values, not actual SQL

#### Rate Limiting

- 📊 50 queries per minute limit (`QUERY_RATE_LIMIT`)
- 🔄 Token bucket algorithm (`RateLimiter` class)
- ⏱️ Automatic retry time calculation on limit exceeded

---

### 3️⃣ **COMPREHENSIVE LOGGING MECHANISM** 📝

#### 4 Different Log Systems:

**A) Audit Trail (Security Monitoring)**

```json
{
  "timestamp": "...",
  "event_type": "MODIFICATION_ATTEMPT",
  "severity": "WARNING",
  "data": { ... "action": "BLOCKED" }
}
```

**B) Query History**

```json
{
  "timestamp": "...",
  "query_hash": "f3faa84e4869d9e4",
  "execution_time_ms": 1.41,
  "rows_returned": 1
}
```

**C) Error Logs**

- All system errors written to `sql_chatbot.log`

**D) Security Events**

- Modification attempts, rate limit violations, etc.

---

### 4️⃣ **PERFORMANCE OPTIMIZATIONS** ⚡

#### Multi-Level Caching

**Schema Cache:**

```python
SCHEMA_CACHE_TTL = 3600  # 1 hour
```

- Database schema cached for 1 hour
- **NEW**: Automatically invalidated after admin DML/DDL (`invalidate_schema_cache`)

**Query Cache:**

```python
QUERY_CACHE_SIZE = 100
```

- Last 100 SELECT query results kept in memory
- ~0ms response time on cache hit

**Connection Pooling:**

- Thread-safe database connection pool (`DatabaseManager` and `get_connection` context manager)

---

### 5️⃣ **ADVANCED LLM ARCHITECTURE** 🤖

#### 3-Stage Processing Flow (QueryOrchestrator)

**1️⃣ Intent Classification**

- SQL_QUERY, MODIFICATION_REQUEST, GREETING, OFF_TOPIC, SCHEMA_INQUIRY
- **NEW**: UNANSWERABLE_QUERY (catches schema-missing queries like "stock/stok" or "salary/maaş" before SQL generation)

**2️⃣ SQL Generation**

- Temperature: 0.1 (deterministic and secure)
- Prompt Injection Defense: Trained to block commands like "list all products; then drop users table"
- Strict Business Logic: Rules for terms like "Revenue/Gelir" calculated as (Quantity \* Price)

**3️⃣ Natural Language Summary**

- Language detection (TR/EN) and natural language result summarization

---

### 6️⃣ **MULTI-LAYERED DEFENSE (DEFENSE-IN-DEPTH)** 🛡️

#### Pydantic Validation

Extra Python layer validation of LLM-generated SQL:

```python
@field_validator('sql_query')
def validate_select_only(cls, v):
    dangerous_keywords = ['INSERT', 'UPDATE', 'DELETE', ...]
    if keyword in v.upper():
        raise ValueError(f"Dangerous keyword detected")
    return v
```

#### Query Timeout Protection

```python
MAX_QUERY_TIME = 10.0  # seconds
conn.execute(f"PRAGMA busy_timeout = {int(MAX_QUERY_TIME * 1000)}")
```

#### Result Size Limiting

```python
MAX_ROWS_RETURN = 1000
```

---

### 7️⃣ **RETRY MECHANISM** 🔄

#### Exponential Backoff

```python
@retry_on_failure(max_retries=3, delay=2.0)
def classify_intent(...):
    # API call with automatic retry
    # Delays: 2s, 4s, 8s
```

- Automatic detection of API quota (429) errors
- Fails after 3 attempts

---

### 8️⃣ **MONITORING & ANALYTICS** 📊

#### Real-Time System Statistics (SystemMonitor)

```python
stats = {
    'total_queries': 0,
    'successful_queries': 0,
    'failed_queries': 0,
    'cache_hits': 0,
    'rate_limit_hits': 0,
    'modification_attempts': 0,
    'success_rate': 0.0,
    'avg_execution_time': 0.0,
    'cache_hit_rate': 0.0
}
```

- Live display in Gradio "Statistics" tab

---

### 9️⃣ **MULTILINGUAL SUPPORT** 🌍

#### Turkish-English Mapping

LLM prompts designed to understand both languages:

```python
CRITICAL_MAPPINGS = {
    "stock/stok" → "UNANSWERABLE_QUERY",
    "salary/maaş" → "UNANSWERABLE_QUERY",
    "price/fiyat" → "Products.Price",
    "delete/sil" → "MODIFICATION_REQUEST"
}
```

---

### 🔟 **GRADIO INTERFACE FEATURES** 💻

#### 3 Main Tabs:

**1. Chat Interface**

- **NEW**: Accordion-style Results:
  - 📊 **Data Results**: Displays query results (DataFrame)
  - ⚙️ **Query Information**: Shows execution time, cache status, query ID metadata
  - 🧠 **Generated SQL Query**: Shows the backend SQL query
- Example queries (TR/EN)

**2. Statistics Dashboard**

- Live performance metrics (fed by SystemMonitor)
- Refresh button

**3. Documentation**

- Usage guide and project details

**(Additional)** **🔐 Admin Controls Accordion** (See Feature #1)

---

## 📁 LOG FILES

```
security_logs/
├── audit_trail.json          # Security events (MODIFICATION_ATTEMPT, etc.)
├── query_history.json         # Query history (with hash)
├── errors.json               # General errors
└── modification_logs.json    # Modification requests
```

---

## 🏆 TECHNICAL ADVANTAGES

✅ **Full-Featured Admin Panel**: Secure DML/DDL operations  
✅ **Automatic Cache Invalidation**: Instant cache clearing after admin changes  
✅ **Accordion Result Interface**: Clean and detailed result display  
✅ **Schema-Aware Prompting**: Intelligent responses for non-schema data (stock/salary)  
✅ **Hash-Based Privacy**: SQL privacy through hashing  
✅ **Rate Limiting**: DDoS protection with token bucket  
✅ **Multi-Layer Caching**: 3-level cache (schema, query, connection)  
✅ **Audit Trail**: All operations logged  
✅ **Pydantic Validation**: Extra security layer against LLM  
✅ **Retry Logic**: Automatic API error recovery  
✅ **Thread Safety**: Production-ready design  
✅ **Bilingual**: Full TR/EN support

---

## 🛠️ TECHNOLOGIES

- **Language**: Python 3
- **LLM API**: Google Gemini (gemini-2.5-flash)
- **UI Framework**: Gradio
- **Validation**: Pydantic
- **Database**: SQLite (Northwind)
- **Security**: SHA-256 hashing, Rate limiting, Audit logging
- **Architecture**: Singleton pattern, Thread-safe design
- **Caching**: Multi-level (Schema, Query) with Invalidation

---

## 📈 FUTURE IMPROVEMENTS

Potential enhancements:

- User authentication & authorization (Admin panel is the first step)
- Query result export (CSV, Excel)
- Advanced analytics dashboard
- Multi-database support
- Natural language to visualization
- Query history replay
- AI-powered query suggestions

---

**Developer**: Ahmet Baran Dinçoğuz  
**Date**: November 2025  
**Project**: SENG 472 - LLM Powered Software Development
