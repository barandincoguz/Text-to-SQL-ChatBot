# LLM-Powered Database Chatbot

---

## 🚀 Overview

This project is an intelligent, LLM-powered chatbot designed to answer natural language questions (in both **Turkish** and **English**) about the Northwind company database.

Instead of a simple "Text-to-SQL" pipeline, this project implements a modular **Router/Agent architecture**. A primary "Router" agent analyzes the user's intent and, if necessary, dispatches the query to a specialized "Text-to-SQL" tool.

The entire backend runs on Python, using the **Gemini API** for reasoning and **Pydantic** for enforcing structured JSON outputs. The user interface is built with **Gradio**.

## 🏗️ Architecture (How it Works)

The application follows a modular "Router-Agent-Tool" pattern for reliability and precision.

1.  **Gradio Interface**: The user types a message (e.g., "stokta kalmayan ürünler?").
2.  **Router Agent (API Call 1)**:
    * The message is sent to `get_user_intent`.
    * A Gemini model (using a `temperature=0.0` and a highly optimized **English** system prompt) analyzes the user's intent against the database schema.
    * It classifies the intent (e.g., `SQL_QUERY`, `GREETING`, `OFF_TOPIC`) by returning a `UserIntent` Pydantic JSON.
3.  **Orchestrator**: The main code checks the intent.
    * If `GREETING` or `OFF_TOPIC`, it returns a hardcoded response.
    * If `SQL_QUERY`, it proceeds to the tools.
4.  **Text-to-SQL Tool (API Call 2)**:
    * The query is sent to `get_sql_from_natural_language`.
    * A second Gemini model (also `temperature=0.0`) converts the question into a **safe, `SELECT-only` SQL query**, returning a `SQLQuery` Pydantic JSON.
5.  **Database Execution**:
    * The `execute_sql_query` function runs the generated SQL on the `northwind.db` (SQLite) file and fetches the raw results (rows and columns).
6.  **Summarizer Tool (API Call 3)**:
    * (Note: This step requires a paid API plan due to quota limits).
    * The raw results, original query, and SQL are sent to `get_final_json_response`.
    * A third Gemini model summarizes the data, replies in the user's original language (TR/EN), and packages everything into the final `FinalResponse` JSON.
7.  **Format & Display**: The `chatbot_response` function parses the `FinalResponse` JSON and displays the answer as clean Markdown in the Gradio chat window.

## ✨ Key Features

* **Natural Language Understanding**: Accepts queries in both Turkish and English.
* **Modular Agent Architecture**: A "Router" (`get_user_intent`) intelligently classifies user intent, preventing unnecessary SQL generation for greetings or off-topic questions.
* **Optimized Prompt Engineering**: All system prompts are in **English** for maximum reliability. The Router prompt is specifically trained to map Turkish terms (like "stok") to the English schema (`UnitsInStock`).
* **Secure & Deterministic SQL**: The Text-to-SQL agent is set to `temperature=0.0` and is forbidden from using `DELETE`, `UPDATE`, or `INSERT` commands.
* **Guaranteed Structured Output**: Uses **Pydantic** models (`UserIntent`, `SQLQuery`, `FinalResponse`) as `response_schema` in the Gemini API calls to ensure all LLM outputs are valid JSON.
* **Interactive UI**: A simple and clean chat interface built with **Gradio**.

## 🛠️ Tech Stack

* **Language**: Python 3
* **LLM API**: Google Gemini (gemini-pro-latest)
* **Chat UI**: Gradio
* **Schema Enforcement**: Pydantic
* **Database**: SQLite
* **Core Libraries**: `google-generativeai`, `gradio`, `pydantic`, `pandas`

## ⚙️ How to Run

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
    cd YourRepoName
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the Database**
    * Download the Northwind SQLite database from [this Wikiversity page](https://en.wikiversity.org/wiki/Database_Examples/Northwind/SQLite).
    * Place the `northwind.db` (or `Northwind.sqlite`, and update `DB_PATH` in the code) file in the root directory of the project.

5.  **Set Your API Key**
    * Get your Gemini API key from [Google AI Studio](https://ai.google.dev/).
    * **IMPORTANT:** Do NOT hardcode your API key in the script. Set it as an environment variable.

    On macOS/Linux:
    ```bash
    export GEMINI_API_KEY="AIza..."
    ```
    On Windows (CMD):
    ```bash
    set GEMINI_API_KEY="AIza..."
    ```

6.  **Run the Application**
    ```bash
    python app.py  # Or whatever you named your main Python file
    ```
    Gradio will start the server and provide a local URL (like `http://127.0.0.1:7860`) to open in your browser.

## ⚠️ API Quota Warning

This project's architecture makes **3 separate API calls** for a single SQL query (1. Intent, 2. SQL, 3. Summary).

The **Gemini Free Tier** is often limited to **2 requests per minute (RPM)**. This will cause the application to fail with a `429 Quota Exceeded` error on the 3rd API call.

**To run this project successfully, you must:**
1.  **Enable billing** on your Google Cloud project to upgrade from the free tier.
2.  (Alternative) If you must use the free tier, you must modify the `run_sql_pipeline` function to **remove the 3rd API call** (`get_final_json_response`) and manually format the `db_data` into the `FinalResponse` object using only Python.
