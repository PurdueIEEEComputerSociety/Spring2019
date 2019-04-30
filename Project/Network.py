import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from time import time

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models

from Loss import *

tf.enable_eager_execution()

#	TODO: There must be a better thing for this
def loadImage(fileName):
	img = Image.open(fileName)
	return kp_image.img_to_array(img)

#	TODO: There must be a better thing for this
def showImage(img):
	#	Make the color profile correct
	img = img.astype('uint8')
	plt.imshow(img)
	plt.show()

def deProcessImage(img):
	#	If the batch dimension exists, remove it
	if (img.shape[0] == 1):
		img = np.squeeze(img, axis=0)
	else:
		img = img.copy()	#	Only need to deep copy if didn't set it from squeeze

	#	Undo the vgg19 preprocessing
	img[:, :, 0] += 103.939
	img[:, :, 1] += 116.779
	img[:, :, 2] += 123.68
	img = img[:, :, ::-1]

	return np.clip(img, 0, 255).astype('uint8')

class StyleTransferModel():
	def __init__(self, numIterations=10):
		#	Layers to be extracted for use
		self.contentLayers = ['block5_conv2'] 
		self.styleLayers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

		self.numIterations = numIterations

		#	Create the model
		self.model = self.createModel(self.contentLayers, self.styleLayers)

	def summary(self):
		self.model.summary()

	def getFeatures(self, image):
		#	The outputs are delimited by the style later
		#	Grab [0] to get rid of batch info
		modelOut = self.model(image)
		styleFeatures = [styleLayer[0] for styleLayer in modelOut[:len(self.styleLayers)]]
		styleFeatures = [gram_matrix(styleFeature) for styleFeature in styleFeatures]
		contentFeatures = [contentLayer[0] for contentLayer in modelOut[len(self.styleLayers):]]
		return styleFeatures, contentFeatures
	
	def setStyleImage(self, fileName):
		self.rawStyleImage = loadImage(fileName)
		#	Give it a batch dimension to that tensorflow/keras will play nicely
		#	Do some preprocessing to scale down the dimensions as desired
		self.styleImage = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(self.rawStyleImage, axis=0))
		self.styleFeatures, _ = self.getFeatures(self.styleImage)
		# self.styleFeatures = [gram_matrix(styleFeature) for styleFeature in self.styleFeatures]

	def setContentImage(self, fileName):
		self.rawContentImage = loadImage(fileName)
		self.contentImage = tf.keras.applications.vgg19.preprocess_input(np.expand_dims(self.rawContentImage, axis=0))
		_, self.contentFeatures = self.getFeatures(self.contentImage)

	def createModel(self, contentLayers, styleLayers):
		vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
		# vgg.trainable = False

		#	Get the outputs of the desired layers
		style_outputs = [vgg.get_layer(name).output for name in contentLayers]
		content_outputs = [vgg.get_layer(name).output for name in styleLayers]
		model_outputs = style_outputs + content_outputs
		
		#	Make new model with vgg input and outputs of desired layers
		return models.Model(vgg.input, model_outputs)

	def transfer(self, alpha=1e3, beta=1e-2):
		#	Initialize to the content image to save time
		x = self.contentImage.copy()
		x = tfe.Variable(x, dtype=tf.float32)

		opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

		norm_means = np.array([103.939, 116.779, 123.68])
		min_vals = -norm_means
		max_vals = 255 - norm_means   

		startTime = time()
		for i in range(self.numIterations):
			print("Iteration {} @ {}s".format(i, time()-startTime))
			grads = self.gradient(alpha, beta, x)

			#	Update the image
			opt.apply_gradients([(grads, x)])
			clipped = tf.clip_by_value(x, min_vals, max_vals)
			x.assign(clipped)

		return deProcessImage(x.numpy())

	def gradient(self, alpha, beta, x):
		#	Record gradients while getting loss
		with tf.GradientTape() as tape:
			x_style, x_content = self.getFeatures(x)

			#	Assumes each layer contributes an equal amount to the style
			#	else would need to put /len(x_style) in the zip and make it more complex
			styleLoss = beta/len(x_style) * sum(get_style_loss(x_s, t_s) for x_s, t_s in zip(x_style, self.styleFeatures))

			contentLoss = alpha/len(x_content) * sum(get_content_loss(x_c, t_c) for x_c, t_c in zip(x_content, self.contentFeatures))

			loss = styleLoss + contentLoss

		return tape.gradient(loss, x)

if __name__ == '__main__':
	contentPath = 'scaledContent/29.jpg'
	stylePath = 'scaledStyle/17.jpg'

	model = StyleTransferModel()

	model.setStyleImage(stylePath)
	model.setContentImage(contentPath)

	styled = model.transfer()
	showImage(styled)


