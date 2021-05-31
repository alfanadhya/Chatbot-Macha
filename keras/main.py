import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
import random
import json
import joblib

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pkl", "rb") as f:
        words, labels, training, output = joblib.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pkl", "wb") as f:
        joblib.dump((words, labels, training, output), f)

ops.reset_default_graph()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8,input_shape=(len(training[0]),)))
model.add(tf.keras.layers.Dense(8,))
model.add(tf.keras.layers.Dense(len(output[0]), activation="softmax"))

try:
    keras.models.load_model("model")
except:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(training, output, epochs=1000, batch_size=8)

    model.save("model")
    model.save_weights("weights.h5")

#####

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


# def chat():
#     print("Start talking with the bot (type quit to stop)!")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == "quit":
#             break

#         results = model.predict([[bag_of_words(inp, words)]])
#         results_index = numpy.argmax(results)
#         tag = labels[results_index]

#         for tg in data["intents"]:
#             if tg['tag'] == tag:
#                 responses = tg['responses']

#         print(random.choice(responses))

# chat()

# import tflearn
# import json
# # import joblib
# from tensorflow import keras
from flask import Flask, request

###
# import nltk
# from nltk.stem.lancaster import LancasterStemmer
# stemmer = LancasterStemmer()
# import random
# import numpy
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

    prediction = model.predict([[bag_of_words(request_json.get('data'), words)]])
    prediction_string = [str(d) for d in prediction]
    response_json = {
    "data" : request_json.get("data"),
    "prediction" : list(prediction_string)
    }

    return json.dumps(response_json)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)