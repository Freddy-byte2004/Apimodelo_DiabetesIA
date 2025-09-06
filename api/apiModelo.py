from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = pickle.load(open('../diabetes_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Recibe JSON
    features = np.array(data['features']).reshape(1, -1)
    probability = model.predict_proba(features)[0][1]  # Clase positiva
    return jsonify({'probabilidad_diabetes': float(probability)})

if __name__ == '__main__':
    app.run(port=5000)