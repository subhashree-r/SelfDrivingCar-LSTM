
# coding: utf-8

# In[ ]:

import argparse
import base64
import json
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

import Queue
 
q = Queue.Queue()

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
#https://www.packtpub.com/mapt/book/application-development/9781785283932/2/ch02lvl1sec26/enhancing-the-contrast-in-an-image
def balance_brightness(image):
	img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
	return img_out
# Fix error with Keras and TensorFlow
import tensorflow as tf
#tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    #firstFrame=0
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # image_array = balance_brightness(image_array)
    image_array = cv2.resize(image_array[60:140,:],(64,64))###my modification
    timeImage=[]
    if firstFrame==0:
        for i in range(10):
            timeImage.append(image_array)
        firstFrame+=1
    else:
        timeImage.pop()
        timeImage.append(image_array)
    
    transformed_image_array=np.zeros((1,10, 64, 64, 3), dtype = np.float32)
    transformed_image_array[0]=timeImage
    print(transformed_image_array.shape)
# This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1)[0][0])
    throttle = float(model.predict(transformed_image_array, batch_size=1)[0][1])
    speed = float(model.predict(transformed_image_array, batch_size=1)[0][2])
    brake = float(model.predict(transformed_image_array, batch_size=1)[0][3])

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    # throttle = 0.32
    # if steering_angle > 0.5:
    # 	throttle = 0.20
    # if 0.15 < steering_angle <= 0.5:
    # 	throttle = 0.25
    # if steering_angle <-0.5:
    # 	throttle = 0.20
    # if -0.5 >= steering_angle > -0.15:
    # 	throttle = 0.25
   #throttle = (26-np.float32(speed))*0.5
    print(steering_angle, throttle)
    send_control(steering_angle, throttle,speed,brake)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle, speed, brake):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle':throttle.__str__(),'speed':speed.__str__(),'brake':brake.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
	
    json_file = open(args.model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    firstFrame=0

    #model.load_weights("model.h5")    
#with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
 #       model = model_from_json(jfile.read())

	
    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

