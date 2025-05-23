# 🌱 SEP4 MAL: Machine Learning for Greenhouse Management

Welcome to the **SEP4 MAL** project! This repository is dedicated to the machine learning (ML) component of our semester project, which focuses on optimizing greenhouse management. The project is divided into two main components:

- **mal-model**: The ML module responsible for training and building a classification model.
- **mal-api**: A REST API that serves the trained model, enabling real-time predictions.

---

## 🚀 Quick Start

### Start the System

To spin up the entire system, including the API and model components, run:

```sh
docker compose -f docker-compose-local.yml up -d --build
```

### Train the Model

To train the machine learning model, execute:

```sh
docker compose -f docker-compose-local.yml --profile build-model up mal-model
```

This will preprocess the data, train the model, and save the necessary artifacts for deployment.

### Stop the System

To stop and clean up the system, use:

```sh
docker compose -f docker-compose-local.yml down
```

---

## 🧠 About the Machine Learning Model

The **mal-model** component is the heart of this project. Here's what it does:

1. **Data Preprocessing**: The model uses `plant_health_data.csv` as its primary dataset. Features are scaled using `scaler_X.save` and `scaler_y.save` to ensure consistent input for the ML algorithm.
2. **Model Training**: A classification model is trained to predict soil moisture levels, which are critical for maintaining plant health in a greenhouse environment.
3. **Model Artifacts**: The trained model is saved as `soil_moisture_model.pkl` for deployment.

### Why Soil Moisture Prediction?
Maintaining optimal soil moisture is essential for plant health and resource efficiency. By predicting soil moisture levels, this model helps automate irrigation systems, reducing water waste and improving crop yield. 🌾

---

## 🌐 mal-api: Serving Predictions

The **mal-api** component exposes the trained model via a REST API. This allows external systems to:

- Send input data for predictions.
- Receive real-time soil moisture predictions.

### Key Features
- **Fast and Scalable**: Built with Python and Flask for high performance.
- **Easy Integration**: Simple endpoints for seamless integration with IoT devices or greenhouse management systems.

---

## 🛠️ Development Setup

### Prerequisites
- Docker and Docker Compose installed on your machine.
- Basic understanding of Python and machine learning.

### Install Dependencies
For local development, navigate to the respective directories and install dependencies:

```sh
# For mal-model
cd src/mal-model
pip install -r requirements.txt

# For mal-api
cd ../mal-api
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
├── docker-compose.yml          # Docker Compose configuration
├── models/                     # Saved model artifacts
│   ├── last_updated.txt        # Timestamp of the last model update
│   ├── scaler_X.save           # Feature scaler
│   ├── scaler_y.save           # Target scaler
│   └── soil_moisture_model.pkl # Trained model
├── src/                        # Source code
│   ├── mal-api/                # REST API for predictions
│   │   ├── Dockerfile
│   │   ├── main.py             # API entry point
│   │   └── prediction_service/ # Prediction logic
│   └── mal-model/              # ML model training
│       ├── Dockerfile
│       ├── main.py             # Training script
│       └── plant_health_data.csv # Dataset
```

---

## 🤝 Contributing

We welcome contributions! Feel free to open issues or submit pull requests to improve the project.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 💡 Acknowledgments

Special thanks to the SEP4 team for their dedication and hard work in making this project a reality. 🌟
