## SunGrid: Solar Irradiance Prediction System

SunGrid is an end-to-end machine learning pipeline for predicting solar irradiance, crucial for optimizing solar energy production. The system centers on a high-fidelity XGBoost regression model served via a production-ready Flask API.

### ✨ Highlights & Performance

| Metric | Value |
|-------:|:-----|
| Model | XGBoost Regressor |
| Accuracy (R²) | 0.9293 (High fidelity) |
| Error (RMSE) | 82.99 W/m² |
| Architecture | Flask REST API & Vanilla JS Dashboard |
| Status | Production Ready |

### 🛠️ Architecture

- Data Pipeline: Feature engineering (log, Box-Cox transformations) on weather and temporal variables.
- Model: XGBoost trained on 32k+ sensor records (September 2016).
- Backend: Flask API (/app.py) for inference, plus endpoints for metrics and health checks.
- Frontend: Interactive dashboard (Chart.js) for real-time visualization of predictions.

### 🚀 Live Demonstration

Explore the interactive dashboard and real-time predictions:
[SunGrid AI Dashboard](https://sungrid-tau.vercel.app/)

### ⚙️ Quick Setup (Development)

Prerequisites: Python 3.9+, pip.

Install dependencies:
```bash
pip install -r backend/requirements.txt
```

Run API server:
```bash
cd backend
python app.py
# Server runs at http://localhost:5000
```

View dashboard:
- Open frontend/index.html in your browser (or serve with a static server).

### 🔗 Contact

For questions or contributions, see the repository files or contact the author.

Built with ☀️ by Shardul
