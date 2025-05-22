from flask import Flask, request, jsonify
from prediction_service.service import predict_with_rf_and_mwa

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    sample = request.json
    forecast = predict_with_rf_and_mwa(sample)
    return jsonify({"forecast": forecast.tolist()})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
