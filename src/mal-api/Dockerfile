FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Copy model files from mal-model directory
COPY ../mal-model/soil_moisture_model.pkl /app/models/
COPY ../mal-model/scaler_X.save /app/models/
COPY ../mal-model/scaler_y.save /app/models/

# Set default environment variable for the port
ENV PORT=5000

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "main.py"]
