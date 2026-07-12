from flask import Flask, render_template, request, jsonify
from waitress import serve
from NeuralNetworkDigits import apiPrediction, load

app = Flask(__name__)

layers, activations = load("Networks/network_layers:784,512,10_acc:0.92.npz")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    prediction = apiPrediction(data, layers, activations)
    return jsonify({"status": "success", "message": f"Data received for data", "prediction": int(prediction)})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)