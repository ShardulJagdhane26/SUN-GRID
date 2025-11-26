# SunGrid - Solar Irradiance Prediction System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌞 Project Overview

SunGrid is an end-to-end machine learning pipeline that predicts solar irradiance using weather and temporal parameters. Built with XGBoost regression achieving **R² = 0.93**, the system includes a production-ready REST API and interactive web dashboard.

### Key Features
- ⚡ **High Accuracy**: R² Score of 0.9293 (RMSE: 82.99, MAE: 33.19)
- 🚀 **Production Ready**: Flask REST API with CORS support
- 📊 **Interactive Dashboard**: Real-time predictions with Chart.js visualization
- 🔄 **Full Pipeline**: Data preprocessing → Feature engineering → Model training → Deployment
- 📦 **32K+ Dataset**: Time-series weather data from September 2016

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **R² Score** | 0.9293 |
| **RMSE** | 82.99 W/m² |
| **MAE** | 33.19 W/m² |
| **Training Samples** | 26,148 |
| **Test Samples** | 6,538 |

---

## 🏗️ Project Structure

```
SunGrid/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── requirements.txt          # Python dependencies
│   ├── model/
│   │   ├── xgboost_model.pkl    # Trained XGBoost model
│   │   └── scaler.pkl           # StandardScaler for features
│   └── notebook/
│       └── SunGrid.ipynb        # Data analysis & model training
├── frontend/
│   └── dashboard.html           # Interactive web interface
├── data/
│   └── solar_irradiance_data.csv
├── README.md
└── .gitignore
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- pip package manager
- Modern web browser

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/sungrid.git
cd sungrid
```

### Step 2: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Verify Model Files
Ensure these files exist in `backend/model/`:
- `xgboost_model.pkl` (trained model)
- `scaler.pkl` (feature scaler)

If missing, run the Jupyter notebook to generate them:
```bash
jupyter notebook notebook/SunGrid.ipynb
# Run all cells, ensure last cells save the model files
```

---

## 🚀 Running the Application

### Backend (Flask API)

```bash
cd backend
python app.py
```

Server will start at: `http://localhost:5000`

**Available Endpoints:**
- `POST /predict` - Get solar irradiance prediction
- `GET /metrics` - View model performance metrics
- `GET /feature_importance` - Feature importance rankings
- `GET /health` - Server health check

### Frontend (Dashboard)

Simply open `frontend/dashboard.html` in your browser, or serve it:

```bash
cd frontend
python -m http.server 8000
# Visit: http://localhost:8000/dashboard.html
```

---

## 📡 API Usage

### Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 48,
    "Pressure": 30.46,
    "Humidity": 59,
    "WindDirection": 177.39,
    "Speed": 1.21,
    "Month": 9,
    "Day": 29,
    "Hour": 23,
    "Minute": 55,
    "Second": 26,
    "risehour": 6,
    "riseminute": 40,
    "sethour": 18,
    "setminute": 50
  }'
```

### Response

```json
{
  "prediction": 245.67,
  "status": "success",
  "unit": "W/m²",
  "model": "XGBoost",
  "input_features": 14
}
```

---

## 🧪 Feature Engineering Pipeline

The model uses sophisticated feature transformations:

1. **Temperature**: Log transformation → `log(T + 1)`
2. **Pressure**: Box-Cox transformation for normality
3. **Humidity**: Box-Cox transformation
4. **Wind Speed**: Log transformation → `log(S + 1)`
5. **Wind Direction**: Min-Max scaling (0-360° → 0-1)
6. **Final Step**: StandardScaler on all 14 features

---

## 📈 Model Training

### Dataset
- **Source**: Solar irradiance sensor data (September 2016)
- **Size**: 32,686 records
- **Features**: 14 engineered features from 11 raw parameters
- **Target**: Solar irradiance (W/m²)

### Algorithm
- **Model**: XGBoost Regressor
- **Parameters**: 
  - `learning_rate`: 0.1
  - `max_depth`: 8
- **Cross-validation**: 80-20 train-test split

### Feature Importance (Top 5)
1. Temperature (28%)
2. Humidity (22%)
3. Hour (19%)
4. Pressure (15%)
5. Month (10%)

---

## 🌐 Deployment

### Option 1: Render.com (Recommended)

1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Render will auto-detect `requirements.txt`
5. Add start command: `gunicorn app:app`
6. Deploy!

### Option 2: Heroku

```bash
heroku create sungrid-api
git push heroku main
heroku open
```

### Option 3: Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

```bash
docker build -t sungrid .
docker run -p 5000:5000 sungrid
```

---

## 🧠 Technical Highlights

### Machine Learning
- XGBoost regression with custom hyperparameters
- Multi-stage feature engineering pipeline
- Statistical transformations (Box-Cox, log)
- StandardScaler for feature normalization

### Backend
- RESTful API design with Flask
- CORS enabled for cross-origin requests
- Error handling and input validation
- Model serialization with joblib

### Frontend
- Vanilla JavaScript (no framework dependencies)
- Chart.js for data visualization
- LocalStorage for prediction history
- Responsive CSS Grid layout

---

## 📝 Future Enhancements

- [ ] Time-series forecasting (LSTM/Prophet)
- [ ] Real-time weather API integration
- [ ] Anomaly detection for sensor errors
- [ ] Model retraining pipeline
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions
- [ ] Database integration (PostgreSQL)
- [ ] User authentication

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- Solar irradiance dataset from [source]
- XGBoost library by DMLC
- Flask web framework
- Chart.js visualization library

---

## 📧 Contact

For questions or feedback, please open an issue or reach out via email.

**Built with ☀️ by [Your Name]**
