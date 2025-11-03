# 🚨 GÜVENLİK UYARISI - API Anahtarı Sızıntısı

## Durum

Tarih: 3 Kasım 2025
Google Gemini API anahtarı yanlışlıkla Git repository'sine commit edildi ve GitHub'a push edildi.

## Açığa Çıkan Anahtar

```
AIzaSyAyfYVvpC6LmEsUYPITFbhJrytsWEC3G9Q
```

## ✅ Alınan Önlemler

### 1. Kod Temizliği

- [x] `hw4.py` dosyasından API anahtarı kaldırıldı
- [x] `project1.py` dosyasından API anahtarı kaldırıldı
- [x] Her iki dosya da artık `os.getenv("GEMINI_API_KEY")` kullanıyor
- [x] `.env.example` dosyası oluşturuldu

### 2. Git İşlemleri

- [ ] Değişiklikler commit edilecek
- [ ] GitHub'a push edilecek

### 3. API Anahtarı Yönetimi

⚠️ **YAPILMASI GEREKENLER:**

1. **HEMEN**: Google AI Studio'ya git ve eski anahtarı sil

   - URL: https://aistudio.google.com/app/apikey
   - Açığa çıkan anahtarı bul ve DELETE butonuna bas

2. **YENİ API ANAHTARI**: Yeni bir anahtar oluştur

   - Aynı sayfada "Create API Key" butonuna tıkla
   - Yeni anahtarı güvenli bir yere kaydet (ör: password manager)

3. **ENVIRONMENT VARIABLE**: Yeni anahtarı ayarla

   ```bash
   # macOS/Linux (.zshrc veya .bash_profile'a ekle)
   export GEMINI_API_KEY="your_new_api_key_here"

   # Veya .env dosyası oluştur
   echo "GEMINI_API_KEY=your_new_api_key_here" > .env
   ```

4. **DOĞRULA**: Uygulamayı test et
   ```bash
   python hw4.py
   # veya
   python project1.py
   ```

## 🔒 Git Geçmişi Temizliği (Opsiyonel ama Önerilen)

Eski commit'lerden API anahtarını tamamen silmek için:

```bash
# git-filter-repo kurulumu (önerilen yöntem)
brew install git-filter-repo

# API anahtarını içeren tüm commit'lerden kaldır
git filter-repo --replace-text <(echo "AIzaSyAyfYVvpC6LmEsUYPITFbhJrytsWEC3G9Q==>***REMOVED***")

# Force push (DİKKAT: Tehlikeli işlem!)
git push origin --force --all
```

**UYARI**: Force push tüm collaborators'ı etkiler. Solo proje ise sorun yok.

## 📚 Gelecek İçin Önlemler

1. **Asla Hardcode Etme**: API anahtarlarını kod içine yazmayın
2. **Environment Variables**: Her zaman `.env` dosyası veya sistem env var kullanın
3. **Git Hooks**: Pre-commit hook ekleyin (API anahtarı kontrolü)
4. **Secret Scanning**: GitHub'ın secret scanning özelliğini aktifleştirin
5. **`.gitignore`**: `.env`, `secrets/`, `*.key` gibi dosyaları ignore edin

## 🔍 Kontrol Listesi

- [x] Kod dosyalarından API anahtarı kaldırıldı
- [x] `.env.example` oluşturuldu
- [ ] Eski API anahtarı Google AI Studio'dan silindi
- [ ] Yeni API anahtarı oluşturuldu
- [ ] Yeni anahtar `.env` dosyasına eklendi
- [ ] `.env` dosyası `.gitignore`'a eklendi
- [ ] Değişiklikler commit edildi
- [ ] GitHub'a push edildi
- [ ] Uygulama test edildi ve çalışıyor

## 📞 İletişim

Sorular için: Proje sahibi ile iletişime geçin.

---

**Son Güncelleme**: 3 Kasım 2025
