# 🧠 RxOrbit – AI-Powered Medical Document Analyzer

RxOrbit is an AI-based system that analyzes **medical prescriptions and lab reports** from images using **OCR + Google Gemini AI** and produces:
- ✅ Structured medical analysis  
- 📑 Downloadable PDF reports  
- 🔊 Audio explanation (text-to-speech)  
- 🤖 Telegram Bot integration  
- 🧪 GRPO-style optimized inference for enhanced output quality  

This project is built as a **mini-project demonstrating real-world AI + NLP + Automation integration**.

---

## 🚀 Key Features

- 📷 **Image-based OCR using Gemini Vision**
- 🧠 **AI-powered medical understanding**
- 🩺 Automatic detection of:
  - Prescription
  - Medical Report
- 📄 **PDF report generation**
- 🔊 **Voice explanation using gTTS**
- 🤖 **Telegram Bot Interface**
- 🧪 **GRPO-style optimization** (multi-sampling + reward-based best output selection)
- 🌐 Multi-language support:
  - English 🇬🇧
  - Kannada 🇮🇳

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Google Gemini API**
- **Telegram Bot API**
- **Pytesseract (OCR fallback)**
- **Pillow (Image Processing)**
- **Streamlit (Web Demo)**
- **gTTS (Audio Output)**
- **ReportLab (PDF Generation)**
- **Git & GitHub (Version Control)**

---

## 📂 Project Structure

```
RxOrbit_Hack/
│
├── app.py
├── telegram_bot.py
├── streamlit_app.py
├── .env
├── requirements.txt
└── README.md
```



---

## ⚙️ Setup Instructions
```
1️⃣ Clone the Repository
git clone <your-repo-url>
cd RxOrbit_Hack

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Create .env File

GEMINI_API_KEY=your_gemini_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token_here

▶️ Running the Applications

Run Telegram Bot
python telegram_bot.py
```
## 🧪 GRPO Optimization (Mini Research Feature)

The project simulates **GRPO (Group Relative Policy Optimization)** by:

- Generating multiple candidate outputs  
- Scoring each candidate using Gemini as a reward model  
- Selecting the best-scoring response  

This improves:

- ✅ Output accuracy  
- ✅ Reduction of hallucinations  
- ✅ Better medical relevance  

---

## 🎯 Use Case Examples

### ✅ Prescription Upload
Upload a prescription image → get:
- Medicine name  
- Dosage  
- Frequency  
- Food timing  

### ✅ Lab Report Upload
Upload a lab report image → get:
- Test values  
- Normal range  
- Status (HIGH / LOW / NORMAL)  

---

## 🔐 Security Note

- API keys are stored using **environment variables**
- Do **NOT** upload `.env` to GitHub  

---

## 📌 Academic Relevance

This project demonstrates:
- Telegram Automation  
- AI-based Medical Document Processing  
- OCR and NLP Pipelines  
- AI Optimization Techniques (GRPO)  
