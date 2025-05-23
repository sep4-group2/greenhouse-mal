FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Copy model files from the models directory in the root context
COPY models/soil_moisture_model.pkl /app/models/
COPY models/scaler_X.save /app/models/
COPY models/scaler_y.save /app/models/

# Set default environment variable for the port
ENV PORT=1337

RUN pip install --no-cache-dir -r /app/src/mal-api/requirements.txt

EXPOSE 1337
CMD ["python", "/app/src/mal-api/main.py"]
