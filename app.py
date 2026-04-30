from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('bmi_model.pkl', 'rb'))

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    height = float(data['height'])
    weight = float(data['weight'])
    gender = int(data['gender'])
    prediction = model.predict([[height, weight, gender]])[0]
    labels = {0:'Extremely Weak',1:'Weak',2:'Normal',3:'Overweight',4:'Obesity',5:'Extreme Obesity'}
    return jsonify({'index': int(prediction), 'category': labels[int(prediction)]})

if __name__ == '__main__':
    app.run(debug=True)