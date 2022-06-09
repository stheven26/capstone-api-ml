#import library
from copyreg import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = __name__(Flask)
model = pickle.load(open("valoai_model.tflite", "rb"))

# routing


@app.route('/predict', methods=["POST"])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({"Prediction": list(prediction)})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
