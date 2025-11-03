# 🔐 API Key Configuration Guide

## Setup

This project now securely loads the API key from a `.env` file.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `google-generativeai`
- `gradio`
- `pydantic`
- `pandas`
- `python-dotenv` ⬅️ NEW!

### 2. Create .env File

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 3. Add Your API Key

Edit the `.env` file and add your API key:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

**Where to Get API Key:**

- https://aistudio.google.com/app/apikey

### 4. Run the Application

```bash
python hw4.py
# or
python project1.py
```

## ✅ Now It Works!

- ✅ API key automatically loads from `.env` file
- ✅ `.env` file is excluded from Git via `.gitignore`
- ✅ Code no longer contains hardcoded secrets
- ✅ Secure and production-ready

## 🔒 Security Notes

- **NEVER** commit the `.env` file to Git
- `.env.example` is template only (no real key)
- Rotate your API keys regularly
- Monitor Google Cloud Console for suspicious activity

---

**Last Updated:** November 3, 2025
