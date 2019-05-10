# celebrity face recognition
from flask import Flask, render_template
from flask import jsonify
from flask import request
from keras.models import model_from_json
import cv2
from model import create_model
from Align import AlignDlib
import dlib
import pickle
from PIL import Image
import base64
import io
import keras
import numpy as np
import tensorflow as tf
import os
import pandas as pd
from sklearn import preprocessing

app = Flask(__name__)

def get_model():
    global model
    json_file = open('saved/sequential_NN_629_model_output_53dim.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("saved/sequential_NN_629_model_ouput_53dim.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Loaded NN model from disk")
    
def get_openface_model():
	global open_face_model
	open_face_model = create_model()
	open_face_model.load_weights('open_face.h5')
	global graph
	graph = tf.get_default_graph()
	print('Loaded openface model')

    
def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]

def load_names_label_encoder():
	global names_encode
	names_encode = preprocessing.LabelEncoder()
	names_encode.classes_ = np.load('saved/names_encode.npy')


def predict_results(encoded_image):
	decoded = base64.b64decode(encoded_image)
	with open("save_image.jpg", 'wb') as f:
		f.write(decoded)
	
	image = load_image("save_image.jpg")
	
	faces = alignment.getAllFaceBoundingBoxes(image)
	response=[]
	
	for i in range(len(faces)):
		face_aligned = alignment.align(96, image, faces[i], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		face_aligned = (face_aligned / 255.).astype(np.float32)
		
		with graph.as_default():
			embedding = open_face_model.predict(np.expand_dims(face_aligned, axis=0))[0]
		with graph.as_default():
			pred = model.predict([[embedding]])
		ind = np.argsort(pred[0])
		print(ind[::-1][:5])
		prediction=[]
		prediction.append(str(names_encode.inverse_transform([ind[::-1][0]])[0])) 
		prediction.append(str(pred[0][ind[::-1][0]]*100))
		response.append({ 'name':prediction[0], 'probability':prediction[1] })
		
	print(response)	
	return jsonify(response)
	
	
def update_new_user(encoded_image,name):
	decoded = base64.b64decode(encoded_image)
	
	num_of_images=0
	if(os.path.isdir("output/"+name)):
		num_of_images = len(os.listdir("output/"+name))
	else:
		os.makedirs("output/"+name)
		
	name_of_file = name+"_"+(str(num_of_images+1))
	with open("output/"+name+"/"+name_of_file+".jpg", 'wb') as f:
		f.write(decoded)
	
	
	


get_model()
get_openface_model()
load_names_label_encoder()
alignment = AlignDlib('models/landmarks.dat')
print("Done !")



@app.route("/predict",methods=["POST"])
def predict():
	msg = request.get_json(force=True)
	encoded_image = msg['image']
	return(predict_results(encoded_image))
	
	
@app.route("/update",methods=["POST"])	
def update():
	msg = request.get_json(force=True)
	encoded_image = msg['image']
	name = msg['name']
	print("New updated image of "+name)
	update_new_user(encoded_image,name)
	return ("RESULT UPDATED")
	










    
    