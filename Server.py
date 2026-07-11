from flask import Flask, render_template, request, jsonify
from waitress import serve
from NeuralNetworkDigits import apiPrediction

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    print(data)
    print(apiPrediction(data))
    return jsonify({"status": "success", "message": f"Data received for data"})

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)