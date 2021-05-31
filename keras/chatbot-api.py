import tflearn
import json
# import joblib
from tensorflow import keras
from flask import Flask, request

###
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import random
import numpy
###

app = Flask(__name__)

# model= joblib.load("model.joblib")
model = keras.models.load_model('model')
model.load_weights("weights.h5")

@app.route("/")
def hello_world():
    return "hello world!"

@app.route("/predict", methods=["POST"])
def predict():
    request_json = request.json
    print("data: {}".format(request_json))
    print("type: {}".format(type(request_json)))

    prediction = model.predict(request_json.get('data'))
    prediction_string = [str(d) for d in prediction]
    response_json = {
    "data" : request_json.get("data"),
    "prediction" : list(prediction_string)
    }

    return json.dumps(response_json)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)





