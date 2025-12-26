# 🏥 RetinalPro AI - Intelligent Retinal Disease Diagnosis System

> AI-powered OCT scan analysis with Google Gemini integration for instant clinical diagnosis

[![Live Demo](https://img.shields.io/badge/Demo-Live-success?style=for-the-badge)](https://retinalpro1.netlify.app)
[![Backend API](https://img.shields.io/badge/API-Active-blue?style=for-the-badge)](https://retinal-pro.onrender.com)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4?style=for-the-badge&logo=google)](https://deepmind.google/technologies/gemini/)

## 🎯 Problem Statement

Manual analysis of OCT (Optical Coherence Tomography) retinal scans is:
- ⏱️ **Time-consuming**: 15-30 minutes per scan
- ❌ **Error-prone**: 8-12% misdiagnosis rate
- 💰 **Expensive**: Requires specialized expertise
- 📉 **Scalability issues**: Limited access in rural areas

## 💡 Our Solution

RetinalPro delivers **instant AI-powered diagnosis** of retinal diseases with:
- ⚡ **Sub-50ms inference time**
- 🎯 **95.6% accuracy** across 4 disease classes
- 🤖 **Google Gemini AI** for clinical report generation
- 📊 **Real-time patient registry** with Firebase Firestore
- 📧 **Automated notifications** for patient follow-ups

---

## 🚀 Live Demo

**🌐 Application:** https://retinalpro1.netlify.app

**Demo Credentials:**
```
Email: taneeshsawant05@gmail.com
Password: [admin@123]
```

⚠️ **First-time users:** Initial login may take 30 seconds (backend warm-up on free tier)

---

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │  Netlify (HTML/JS/Tailwind)
│   (Client)  │  
└──────┬──────┘
       │
       │ HTTPS
       ▼
┌─────────────────────────────────────┐
│     Backend API (Flask/Python)      │
│     Render Cloud Platform           │
│  ┌───────────────────────────────┐  │
│  │  CNN Model (TensorFlow)       │  │
│  │  OCT Image Classification     │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │  Google Gemini AI  ⭐         │  │
│  │  Clinical Report Generation   │  │
│  └───────────┬───────────────────┘  │
│              ▼                       │
│  ┌───────────────────────────────┐  │
│  │  Firebase Services            │  │
│  │  • Authentication             │  │
│  │  • Firestore Database         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## 🔥 Key Features

### 🧠 AI-Powered Diagnosis
- Custom CNN trained on 84,000+ OCT images
- Detects: **CNV, DME, DRUSEN, NORMAL**
- Edge inference mode for resource-constrained deployment
- Real-time confidence scoring

### 🤖 Google Gemini Integration
- **Gemini 2.0 Flash Experimental**: Clinical report generation
- **Gemini 2.5 Flash**: AI medical chat assistant
- Contextual explanations in natural language
- Evidence-based recommendations

### 🔐 Secure Clinical Workflow
- Firebase Authentication for clinician access
- Firestore real-time patient registry
- Role-based access control
- HIPAA-compliant data handling

### 📧 Smart Notifications
- SendGrid email integration
- Automated patient follow-up alerts
- Severity-based scheduling
- Customizable notification templates

### 📅 Intelligent Scheduling
- FullCalendar integration
- Surgery/appointment management
- Priority-based scheduling
- Team collaboration features

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: Vanilla JavaScript
- **Styling**: TailwindCSS
- **Icons**: Lucide Icons
- **Calendar**: FullCalendar
- **Hosting**: Netlify

### Backend
- **Language**: Python 3.9+
- **Framework**: Flask
- **ML/AI**: TensorFlow, Keras
- **Deployment**: Render

### Google Technologies ⭐
- **Google Gemini AI** (Primary)
  - `gemini-2.0-flash-exp`
  - `gemini-2.5-flash`
- **Firebase Authentication**
- **Cloud Firestore**

### Additional Services
- SendGrid (Email)
- UptimeRobot (Monitoring)
- NumPy, Pillow (Image Processing)

---

## 📦 Installation & Setup

### Prerequisites
```bash
python >= 3.9
pip
node (optional, for local frontend testing)
```

### Backend Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/retinal-pro.git
cd retinal-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key
FIREBASE_KEY_PATH=path/to/firebase-key.json
SENDGRID_API_KEY=your_sendgrid_key
GMAIL_USER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_app_password
MODEL_PATH=models/retinal_cnn_model.h5
```

4. **Download model (optional)**
```bash
# Place your trained model in models/ directory
# Or the system will use edge inference mode
```

5. **Run the backend**
```bash
python app.py
```

Backend will start at: `http://localhost:5000`

### Frontend Setup

1. **Update API endpoints** (if testing locally)
In `index.html`, update:
```javascript
const API_BASE_URL = 'http://localhost:5000';
```

2. **Open in browser**
```bash
# Simply open index.html in your browser
# Or use a local server:
python -m http.server 3000
```

---

## 🎮 Usage Guide

### For Clinicians

1. **Login**
   - Navigate to the application
   - Enter credentials
   - Access clinical dashboard

2. **Analyze OCT Scan**
   - Click "Scan Engine"
   - Enter patient details
   - Upload OCT image (JPG/PNG)
   - Click "Run Neural Inference"
   - Review AI-generated diagnosis and report

3. **Manage Patient Registry**
   - View all patient records
   - Search by name/ID/diagnosis
   - Send automated follow-up emails
   - Export data as CSV

4. **Schedule Appointments**
   - Access "Planner Hub"
   - Create new appointments/surgeries
   - View calendar overview
   - Set priorities and reminders

5. **AI Assistant**
   - Ask medical queries in chat
   - Get instant evidence-based responses
   - Context-aware recommendations

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.6% |
| **Inference Time** | <50ms |
| **Model Size** | ~15MB |
| **Input Size** | 224x224x3 |
| **Classes** | 4 (CNV, DME, DRUSEN, NORMAL) |

### Disease Classes

- **CNV**: Choroidal Neovascularization
- **DME**: Diabetic Macular Edema
- **DRUSEN**: Drusen deposits (early AMD)
- **NORMAL**: Healthy retina

---

## 🔌 API Endpoints

### Health Check
```bash
GET /health
Response: {"status": "online", "cnn_status": "full_model"}
```

### Predict Diagnosis
```bash
POST /predict
Content-Type: multipart/form-data

Body:
- image: OCT scan file
- patientName: string
- pati