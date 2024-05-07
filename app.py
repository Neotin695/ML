import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

from model import encoder

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route('/', methods=['POST'])
def process_array():


    data = request.get_json()

    if 'array' not in data:
        return jsonify({"error": "No array key found in JSON"}), 400

    # Get the array from the JSON data
    array_data = data['array']
    print(len(array_data))
    features = np.array([array_data])
    predicted_labels = model.predict(features)

    # يمكنك طباعة التصنيفات المتوقعة للتحقق منها
    print("Predicted Labels:", predicted_labels)
    predicted_labels_names = encoder.inverse_transform(predicted_labels)

    # طباعة أسماء الفئات المتوقعة
    print("Predicted Labels Names:", predicted_labels_names)
    result = predicted_labels_names.tolist()
    # Return the processed array as JSON
    return json.dumps(result)
if __name__ == "__main__":
    flask_app.run(host='127.0.0.1',port=5000)