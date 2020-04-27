from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import cv2
import os
import numpy as np
from flask_cors import CORS, cross_origin
import tensorflow.keras
from PIL import Image, ImageOps
import base64
import json
import dlib
import imutils
from imutils import face_utils

handLabels = ["Stretched", "NotStretched"]
faceLabels = ["MildPain", "NoPain"]
facialLabels = ["No Face Droop", "Face Droop"]

model_face = tensorflow.keras.models.load_model('keras_face_model.h5')
model_hand = tensorflow.keras.models.load_model('keras_hand_model.h5')

model_facial_path = tensorflow.keras.models.load_model('Model.h5')


# Process image and predict label
def processImgFacial(IMG_PATH):
    global shape
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    image = cv2.imread(IMG_PATH)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    coord = []
    print(rects)
    if len(rects) > 0:
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            i = 0
            for (x, y) in shape:
                if i > 36:
                    coord.append(x)
                    coord.append(y)
                i += 1

    t2 = np.array([coord])
    normalized_image_array = (t2.astype(np.float32) / 127.0) - 1

    model_facial_path.load_weights('weight.h5')
    prediction = model_facial_path.predict(normalized_image_array)
    print("pred", prediction)

    lastfacialLabel = facialLabels[np.argmax(np.squeeze(prediction[0]))]

    print(lastfacialLabel)
    confidence = np.max(np.squeeze(prediction))

    writeList = [str(confidence), lastfacialLabel]
    with open('facialdroop.txt', 'w') as filehandle:
        json.dump(writeList, filehandle)
    return lastfacialLabel


# Process image and predict label
def processImgFace(IMG_PATH):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(IMG_PATH)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model_face.predict(data)
    print(prediction)
    lastpainLabel = faceLabels[np.argmax(np.squeeze(prediction))]
    confidence = np.max(np.squeeze(prediction))
    writeList = [str(confidence), lastpainLabel]
    with open('face.txt', 'w') as filehandle:
        json.dump(writeList, filehandle)
    return lastpainLabel


# Process image and predict label
def processImgHand(IMG_PATH):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(IMG_PATH)
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model_hand.predict(data)
    lasthandLabel = handLabels[np.argmax(np.squeeze(prediction))]
    confidence = np.max(np.squeeze(prediction))
    writeList = [str(confidence), lasthandLabel]
    with open('hand.txt', 'w') as filehandle:
        json.dump(writeList, filehandle)
    return lasthandLabel


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def main():
    return """
        Application is working
    """


# About page with render template
@app.route("/about")
def postsPage():
    return render_template("about.html")


@app.route("/analysisreport", methods=["POST"])
def resultPage():
    # open output file for reading
    with open('face.txt', 'r') as filehandle:
        faceResult = json.load(filehandle)
    # open output file for reading
    with open('hand.txt', 'r') as filehandle:
        handResult = json.load(filehandle)
    with open('facialdroop.txt', 'r') as filehandle:
        FacialDroop = json.load(filehandle)

    dictRecult = {}
    dictRecult["hand_lbl"] = handResult[1]
    dictRecult["face_lbl"] = faceResult[1]
    dictRecult["facial_lbl"] = FacialDroop[1]

    dictRecult["hand_acc"] = str(round(float(handResult[0]) * 100, 2))
    dictRecult["face_acc"] = str(round(float(faceResult[0]) * 100, 2))
    dictRecult["facial_acc"] = str(round(float(FacialDroop[0]) * 100, 2))
    app_json = json.dumps(dictRecult)
    return app_json


@app.route("/processfacial", methods=["POST"])
def processReqFacial():
    if request.user_agent.browser is None:
        data = request.files["img"]
        data.save("temp.jpg")
    else:
        data = request.form["photo"]
        data = data.split(",")[1]
        buff = np.fromstring(base64.b64decode(data), np.uint8)
        data = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        im = Image.fromarray(data)
        im.save("temp.jpg")
    resp = processImgFacial("temp.jpg")
    return resp


@app.route("/processface", methods=["POST"])
def processReqFace():
    if request.user_agent.browser is None:
        data = request.files["img"]
        data.save("temp.jpg")
    else:
        data = request.form["photo"]
        data = data.split(",")[1]
        buff = np.fromstring(base64.b64decode(data), np.uint8)
        data = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        im = Image.fromarray(data)
        im.save("temp.jpg")
    resp = processImgFace("temp.jpg")
    return resp


@app.route("/processhand", methods=["POST"])
def processReqHand():
    if request.user_agent.browser is None:
        data = request.files["img"]
        data.save("temp.jpg")
    else:
        data = request.form["photo"]
        data = data.split(",")[1]
        buff = np.fromstring(base64.b64decode(data), np.uint8)
        data = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        im = Image.fromarray(data)
        im.save("temp.jpg")
    resp = processImgHand("temp.jpg")
    return resp


if __name__ == "__main__":
    app.run(debug=True)
