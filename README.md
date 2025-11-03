# 🚀 Enterprise Text-to-SQL Chatbot System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio](https://img.shields.io/badge/Gradio-Latest-orange)](https://gradio.app/)

---

## � Overview

An enterprise-grade, LLM-powered chatbot system designed to answer natural language questions (in both **Turkish** and **English**) about the Northwind database. This project features production-ready security, comprehensive logging, admin controls, and advanced caching mechanisms.

### Key Highlights

- 🔒 **Enterprise Security**: Multi-layered defense with rate limiting, audit logging, and SQL injection protection
- 👨‍💼 **Admin Panel**: Password-protected DML/DDL operations for authorized users
- ⚡ **High Performance**: Multi-level caching with automatic invalidation
- 📊 **Comprehensive Monitoring**: Real-time statistics and audit trails
- 🌍 **Bilingual**: Full support for English and Turkish queries

---

## 🏗️ Architecture

The application follows a modular **Router-Agent-Tool** pattern:

```
User Query → Intent Classification → SQL Generation → Database Execution → Result Formatting
     ↓              (LLM Call 1)        (LLM Call 2)      (SQLite)         (LLM Call 3)
  Gradio UI      UserIntent JSON      SQLQuery JSON     Raw Results      FinalResponse JSON
```

### Components

1. **Router Agent**: Classifies user intent (SQL_QUERY, GREETING, MODIFICATION_REQUEST, etc.)
2. **SQL Generator**: Converts natural language to safe SELECT queries
3. **Database Manager**: Thread-safe connection pooling with query caching
4. **Security Auditor**: Logs all operations and blocks unauthorized modifications
5. **Admin Panel**: Secure interface for authorized database modifications

---

## ✨ Features

### 🔐 Security

- **Read-Only Access**: Standard users can only execute SELECT queries
- **Admin Controls**: Password-protected panel for DML/DDL operations
- **Rate Limiting**: Token bucket algorithm (50 queries/minute)
- **Audit Trail**: All queries and security events logged with SHA-256 hashes
- **Pydantic Validation**: Extra layer of SQL query validation

### ⚡ Performance

- **Schema Caching**: 1-hour TTL with automatic invalidation
- **Query Caching**: LRU cache for last 100 queries (~0ms response on hit)
- **Connection Pooling**: Thread-safe database connection management
- **Retry Logic**: Exponential backoff for API failures

### 📊 Monitoring

- **Real-Time Statistics**: Query count, success rate, execution times
- **Comprehensive Logging**: Audit trail, query history, error logs
- **Security Events**: Modification attempts, rate limit violations

### 🌐 User Interface

- **Accordion Results**: Organized display with data, metadata, and SQL
- **Example Queries**: Pre-loaded examples in both languages
- **Live Statistics**: Refreshable performance metrics
- **/sql Command**: Admin quick-execute from chat

---

## 🛠️ Tech Stack

| Component    | Technology                         |
| ------------ | ---------------------------------- |
| Language     | Python 3.8+                        |
| LLM API      | Google Gemini (gemini-2.5-flash)   |
| UI Framework | Gradio                             |
| Validation   | Pydantic                           |
| Database     | SQLite (Northwind)                 |
| Security     | SHA-256, Rate Limiting, Audit Logs |
| Architecture | Singleton, Thread-Safe Design      |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/barandincoguz/-Text-to-SQL-ChatBot.git
cd -Text-to-SQL-ChatBot
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

**Get API Key**: [Google AI Studio](https://aistudio.google.com/app/apikey)

### 5. Run the Application

```bash
python hw4.py
```

The Gradio interface will launch at `http://127.0.0.1:7860`

---

## 🔑 Admin Access

**Default Admin Password**: `admin123`

**Admin Capabilities**:

- Execute UPDATE, INSERT, DELETE commands
- Create new tables
- Run SELECT queries via `/sql` command
- Real-time schema cache invalidation

**Security**: DROP, TRUNCATE, ALTER, VACUUM are blocked even for admins.

---

## 📂 Project Structure

```
.
├── hw4.py                    # Main application file
├── project1.py               # Alternative version
├── requirements.txt          # Python dependencies
├── .env                      # API keys (gitignored)
├── .env.example             # Template for .env
├── northwind.db             # SQLite database
├── security_logs/           # Audit and query logs
│   ├── audit_trail.json
│   ├── query_history.json
│   └── errors.json
└── README.md                # This file
```

---

## 📊 Usage Examples

### English Queries

```
"How many customers do I have?"
"Show top 5 products by price"
"List all orders from 2024"
```

### Turkish Queries

```
"Kaç müşterim var?"
"En pahalı 5 ürünü göster"
"2024 yılındaki tüm siparişleri listele"
```

### Admin Commands (with password)

```
/sql UPDATE Products SET Price = 150 WHERE ProductID = 1;
/sql INSERT INTO Categories (CategoryName) VALUES ('New Category');
```

---

## ⚠️ Important Notes

### API Quota

- Free tier: ~2 requests/minute
- This app makes 3 LLM calls per query
- **Recommended**: Enable billing for production use

### Security

- Never commit `.env` file
- Rotate API keys regularly
- Monitor `security_logs/` for suspicious activity
- Review admin access logs

---

## 📈 Performance Metrics

From actual production logs:

- **Average Query Time**: ~1.5ms
- **Cache Hit Rate**: High (same queries return in 0ms)
- **Security**: 6+ blocked modification attempts
- **Total Queries**: 14+ successful executions

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ahmet Baran Dinçoğuz**  
SENG 472 - LLM Powered Software Development  
November 2025

---

## 📚 Additional Documentation

- [Project Description](ProjectDescription.md) - Detailed technical overview
- [API Setup Guide](API_SETUP.md) - API key configuration
- [Security Incident Report](SECURITY_INCIDENT.md) - Security procedures

---

**Last Updated**: November 3, 2025
