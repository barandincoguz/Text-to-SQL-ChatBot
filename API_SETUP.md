# 🔐 API Anahtarı Yapılandırması

## Kurulum

Bu proje artık `.env` dosyasından API anahtarını güvenli bir şekilde yükler.

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

Bu komut şunları yükler:

- `google-generativeai`
- `gradio`
- `pydantic`
- `pandas`
- `python-dotenv` ⬅️ YENİ!

### 2. .env Dosyası Oluştur

`.env.example` dosyasını `.env` olarak kopyalayın:

```bash
cp .env.example .env
```

### 3. API Anahtarınızı Ekleyin

`.env` dosyasını düzenleyin ve API anahtarınızı ekleyin:

```bash
GEMINI_API_KEY=your_actual_api_key_here
```

**Nereden API Anahtarı Alınır:**

- https://aistudio.google.com/app/apikey

### 4. Uygulamayı Çalıştırın

```bash
python hw4.py
# veya
python project1.py
```

## ✅ Artık Çalışıyor!

- ✅ API anahtarı `.env` dosyasından otomatik yüklenir
- ✅ `.env` dosyası `.gitignore` ile Git'ten hariç tutulur
- ✅ Kod artık hardcoded secret içermiyor
- ✅ Güvenli ve production-ready

## 🔒 Güvenlik Notları

- `.env` dosyasını **ASLA** Git'e commit etmeyin
- `.env.example` sadece şablon içindir (gerçek anahtar yok)
- API anahtarlarınızı düzenli olarak yenileyin
- Şüpheli aktivite için Google Cloud Console'u kontrol edin

---

**Son Güncelleme:** 3 Kasım 2025
