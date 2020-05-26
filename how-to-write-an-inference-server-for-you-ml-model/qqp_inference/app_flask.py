from flask import Flask, request, jsonify
from qqp_inference.model import PythonPredictor


app = Flask(__name__)
predictor = PythonPredictor.create_for_demo()


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json
    result = predictor.predict(payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
