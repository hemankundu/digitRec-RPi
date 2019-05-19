import tensorflow as tf
import keras
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
import cv2
import json
import capture

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


img_w = img_h = 28
done = False
while not done:
	capture.capture(0, True, (280, 280))
	input_str = input('File name [input.png]/exit/config : ')
	if input_str == '':
		input_str = 'input.png'
	elif input_str == 'exit':
		print('exiting..')
		break
	elif input_str == 'config':
		with open('config.json', 'r') as f:
			config = json.load(f)
			print('Current configuration: ', config)
		print('Edit configuration:')
		config = {'medianBlur':0.0, 'blockSize1':0.0, 'C1':0.0,'blockSize2':0.0, 'C2':0.0, 'cropx10':1.0}
		config_keys = list(config.keys())
		config_keys.sort()
		print(config_keys)
		val = list(map(float, input().split()))
		for i in range(len(config_keys)):
			config[config_keys[i]] = val[i]
		with open('config.json', 'w') as f:
			config = json.dump(config, f)
			print('Configuration saved')
			print('Current configuration: ', config)

	elif '.' not in input_str:
		input_str += '.png'
	with open('config.json', 'r') as f:
			config = json.load(f)
			print('Configuration loaded')
	frame = cv2.imread('input.png')
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.medianBlur(frame, int(config['medianBlur']))
	frame = cv2.adaptiveThreshold(frame,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY\
		,int(config['blockSize1']),int(config['C1']))
	kernel = np.ones((3, 3), np.uint8)
	frame = cv2.dilate(frame, kernel, iterations=1)
	height, width  = frame.shape
	if height > width:
		x = 0
		y = (height-width)//2
	elif width > height:
		x = (width-height)//2
		y = 0
	crop = int((height//10)*config['cropx10'])
	frame = frame[crop+y : height-crop-y, crop+x : width-crop-x]
	plt.imshow(frame)

	frame = cv2.resize(frame, (img_w, img_h), interpolation = cv2.INTER_AREA)
	frame = cv2.adaptiveThreshold(frame,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY\
		,int(config['blockSize2']),int(config['C2']))
	frame = cv2.bitwise_not(frame)
	plt.imshow(frame)
	frame = np.array(frame).reshape(1, 28, 28, 1)
	#print(frame)
	frame = tf.keras.utils.normalize(frame, axis=0)


	prediction = loaded_model.predict(frame)
	#print(prediction)

	prediction = prediction[0]
	bestclass = ''
	bestconf = -1
	for n in [0,1,2,3,4,5,6,7,8,9]:
		if (prediction[n] > bestconf):
			bestclass = str(n)
			bestconf = prediction[n]
	print('\nI think this digit is a ' + bestclass + ' with ' + str(bestconf * 100) + '% confidence.')
	while True:
		in_str = input('[continue]/exit: ')
		if in_str == '' or in_str == 'continue':
			break
		elif in_str == 'exit':
			print('exiting..')
			done = True
			break
	
