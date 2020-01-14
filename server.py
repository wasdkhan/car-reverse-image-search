#!/usr/bin/env python

# import libraries
print('importing libraries')

from flask import Flask, request, render_template, flash, redirect, url_for, session, jsonify
from werkzeug import secure_filename
import logging
import random
import time

from PIL import Image
import requests, os
from io import BytesIO

# import pytorch and models
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import resnet_50

# import indexing
import faiss
import numpy as np
import os

# import settings
from settings import *

# import faiss index
print('done!\nloading index...')

if os.path.exists(index_filename):
	index = faiss.read_index(index_filename)
else:
	print('index not found. loading training features...')
	# load feature vectors and get shape number of vectors (n) x dimension (2048)
	train_features = np.load(train_features_filename)
	(n, dimension) = train_features.shape

	print('done!\ntraining index...')
	index = faiss.IndexHNSWFlat(dimension, 32)
	index.verbose = True
	index.add(train_features)
	index.hnsw.efSearch = 16
	print('done!\nsaving index to file: ', index_filename)
	faiss.write_index(index, index_filename)

# import models
print('done!\nloading up saved model...')

device = torch.device('cpu')

# load trained resnet model
model = resnet_50.resnet_50
model.load_state_dict(torch.load('resnet_50.pth'))

# remove last layer (FC) and do not require gradient, set to eval mode
modules = list(model.children())[:-1]
model = nn.Sequential(*modules)
for p in model.parameters():
	p.requires_grad = False
model.to(device)
model.eval()

# pre-processing transforms
normalize = transforms.Normalize(
	mean=[0.485, 0.456, 0.406],
	std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	normalize
])

def classify_image(img, imgpath=""):
	app.logger.info("Classifying image %s" % (imgpath),)
	t = time.time() # get execution time

	img_tensor = preprocess(img).unsqueeze(0)
	features_var = model(Variable(img_tensor)).data.numpy()
	num_input_features = len(features_var)
	k = 5

	_, I = index.search(features_var, k)
	for i in range(num_input_features):
		suggested_labels = []
		suggested_filenames = []
		for idx in I[i]:
			suggested_filenames.append(train_labels[idx])
			suggested_label = train_labels[idx].split('/')[0]
			suggested_labels.append(suggested_label)
		print("suggested_labels", suggested_labels)
	pred_class = suggested_labels[0]

	dt = time.time() - t
	app.logger.info("Execution time: %0.02f seconds" % (dt))
	app.logger.info("Image %s classified as %s" % (imgpath, pred_class))

	return (suggested_labels, suggested_filenames)

print('done!\nlaunching the server...')

# set flask params

def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'random secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		url = request.form['url'] if 'url' in request.form else ''
		if url:
			session['url'] = url
			session.pop('imgpath') if 'imgpath' in session.keys() else None
			return redirect(url_for('predict'))
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No seleced file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			imgpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(imgpath)
			#img = Image.open(imgpath).convert('RGB')
			session['imgpath'] = imgpath
			session.pop('url') if 'url' in session.keys() else None
			return redirect(url_for('predict'))
	return '''
	<!DOCTYPE html>
	<title>Cars Classification</title>
	<h1>Cars Classification</h1>
	<h2>Upload new file</h2>
	<form method=post enctype=multipart/form-data>
		Image File:<input type=file name=file>
		<input type=submit value=Upload>
	</form>
	<h2>OR Paste URL:</h2>
	<form method=post>
		URL:<input type=url name=url>
		<input type=submit value=Go>
	</form>
	'''

@app.route('/predict')
def predict():
	session_keys = list(session.keys())
	url = session['url'] if 'url' in session_keys else ''
	imgpath = session['imgpath'] if 'imgpath' in session_keys else ''
	if url:
		imgpath = url
		response = requests.get(url)
		img = Image.open(BytesIO(response.content)).convert('RGB')
	elif imgpath:
		imgpath = session['imgpath']
		img = Image.open(imgpath).convert('RGB')
	else:
		imgpath = 'static/uploads/rav4.jpg'
		img = Image.open(imgpath).convert('RGB')
	(suggested_labels, suggested_filenames) = classify_image(img, imgpath=imgpath)

	return render_template('predict.html', suggested_labels=suggested_labels, imgpath=imgpath, suggested_filenames=suggested_filenames)

@app.route('/predict2', methods=['GET'])
def predict2():
	url = request.args['url']
	app.logger.info("Classifying image %s" % (url),)

	response = requests.get(url)

	img = Image.open(BytesIO(response.content)).convert('RGB')

	(suggested_labels, suggested_filenames) = classify_image(img, imgpath=url)

	# return jsonify(suggested_labels)
	return render_template('predict.html', suggested_labels=suggested_labels, imgpath=url, suggested_filenames=suggested_filenames) 

if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=True, port=PORT)
	#app.run(host="0.0.0.0", debug=False, port=PORT)
