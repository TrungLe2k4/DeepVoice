# backend_flask/app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# 🧠 Nạp mô hình đã huấn luyện (vd: XGBoost)
model = joblib.load("Models/xgb_model.pkl")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    features = np.array(data.get("features", []), dtype=float).reshape(1, -1)

    # Dự đoán xác suất deepfake
    prob = float(model.predict_proba(features)[0, 1])
    reason = "MFCC đặc trưng bất thường"  # sau có thể thêm logic explain

    return jsonify({"prob": prob, "reason": reason})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
