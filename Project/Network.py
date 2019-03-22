from pprint import pprint as pp
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from time import time

from keras.datasets import mnist
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape

NUM_EPOCHS		= 20
BATCH_SIZE		= 70

STYLES_DIR		= "scaledStyle"
CONTENT_DIR		= "scaledContent"

def getData(styleDir=STYLES_DIR, contentDir=CONTENT_DIR):

	styles = np.array([plt.imread(f)[:,:] for f in glob("{}/*".format(styleDir))])
	content = np.array([plt.imread(f)[:,:] for f in glob("{}/*".format(contentDir))])

	#	Scale to [0,1]
	styles = styles.astype('float32')/float(styles.max())
	content = content.astype('float32')/float(content.max())

	return (styles, content)

def getModel(x_train, x_test):

	enc_in = Input(shape=x_train.shape[1:])
	enc_c1 = Conv2D(16, (3,3), activation='relu', padding='same')(enc_in)
	enc_p1 = MaxPooling2D((2,2), padding='same')(enc_c1)
	enc_c2 = Conv2D(8, (3,3), activation='relu', padding='same')(enc_p1)
	enc_p2 = MaxPooling2D((2,2), padding='same')(enc_c2)
	enc_c3 = Conv2D(8, (3,3), activation='relu', padding='same')(enc_p2)
	enc_p3 = MaxPooling2D((2,2), padding='same')(enc_c3)
	enc_ls = Reshape((4*4*8,))(enc_p3)

	encoder = Model(inputs=enc_in, outputs=enc_ls, name='encoder')
	# encoder.summary()

	dec_in = Input((4*4*8,))
	dec_rs = Reshape((4,4,8))(dec_in)
	dec_c1 = Conv2D(8, (3,3), activation='relu', padding='same')(dec_rs)
	dec_u1 = UpSampling2D((2,2))(dec_c1)
	dec_c2 = Conv2D(8, (3,3), activation='relu', padding='same')(dec_u1)
	dec_u2 = UpSampling2D((2,2))(dec_c2)
	dec_c3 = Conv2D(16, (3,3), activation='relu')(dec_u2)
	dec_u3 = UpSampling2D((2,2))(dec_c3)
	dec_op = Conv2D(1, (3,3), activation='sigmoid', padding='same')(dec_u3)

	decoder = Model(inputs=dec_in, outputs=dec_op, name='decoder')
	# decoder.summary()

	autoencoder_output = decoder(encoder(enc_in))
	autoencoder = Model(enc_in, autoencoder_output)
	# autoencoder.summary()

	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	autoencoder.fit(x_train, x_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test,x_test))	

	return (encoder, decoder)

def saveModels(models):
	for model in models:
		with open('{}.json'.format(model.name), 'w') as file:
			file.write(model.to_json())
		model.save_weights('{}.h5'.format(model.name))

def loadModels(modelNames):
	retval = []
	for ind, name in enumerate(modelNames):
		with open(name+".json", 'r') as file:
			retval.append(model_from_json(file.read()))
		retval[ind].load_weights(name+".h5")
	return retval

if __name__ == '__main__':
	TRAIN = False

	elapsed = -time()
	styles, content = getData()
	elapsed += time()
	print("Time to read data: {}".format(elapsed))

	# if TRAIN:
	# 	encoder, decoder = getModel(train, test)
	# 	saveModels((encoder, decoder))
	# else:
	# 	encoder, decoder = loadModels(('encoder', 'decoder'))
	# 	encoder.summary()
	# 	decoder.summary()




	# digits = [test[np.where(test_labels==i)[0][0],:,:] for i in range(10)]
	# digits[5] = test[np.where(test_labels==5)[0][1],:,:]	#	The first 5 is disgusting

	# _, ax = plt.subplots(2, 5, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(5):
	# 		ax[i,j].imshow(digits[i*5 + j][:,:,0])
	# # plt.show()
	# plt.savefig("GroundTruth.png")
	# plt.clf()

	# latent_digits = encoder.predict(np.array(digits))

	# latent_distances = [[np.linalg.norm(i-j) for j in latent_digits] for i in latent_digits]
	# for d in latent_distances:
	# 	print(d)

	# latent_noise = np.random.rand(10, 128)	#	Generate 10 random noise vectors

	# print()
	# latent_distances = [[np.linalg.norm(i-j) for j in latent_noise] for i in latent_noise]
	# for d in latent_distances:
	# 	print(d)

	# styles = [plt.imread(f)[:,:] for f in glob("Styles/*_s.png")]

	# _, ax = plt.subplots(2, 3, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(3):
	# 		ax[i,j].imshow(styles[i*3 + j][:,:,0])
	# # plt.show()
	# plt.savefig("Styles.png")
	# plt.clf()

	# latent_styles = encoder.predict(np.array(styles)[:,:,:,0].reshape(6,28,28,1))

	# latent_experiments = np.array([latent_digits[7] - latent_digits[1],				#	7-1 = top bar?
	# 							   latent_digits[9] - latent_digits[1],				#	9-1 = 0?
	# 							   latent_digits[0] + latent_digits[1],				#	0+1=9?
	# 							   0.6*latent_digits[0] + 0.55*latent_digits[1],		#	scale first
	# 							   latent_digits[3] * 2,
	# 							   latent_digits[5] * 2,
	# 							   latent_digits[5] + 0.1*latent_digits[0],			#	Does this make the 5 curvier?
	# 							   latent_digits[7] + 0.1*latent_digits[0],			#	Does this make the 7 curvier?
	# 							   latent_digits[8] + 0.1*latent_digits[7],			#	Does this make the 8 stabbier?
	# 							   latent_digits[2] + latent_noise[0]				#	Does adding noise do anything?
	# 								])


	# reconstructed_digits = decoder.predict(latent_digits)
	# reconstructed_noise = decoder.predict(latent_noise)
	# reconstructed_exps = decoder.predict(latent_experiments)
	# reconstructed_styles = decoder.predict(latent_styles)


	# _, ax = plt.subplots(2, 5, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(5):
	# 		ax[i,j].imshow(reconstructed_digits[i*5 + j][:,:,0])
	# plt.savefig("Reconstruction.png")
	# plt.clf()	

	# _, ax = plt.subplots(2, 5, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(5):
	# 		ax[i,j].imshow(reconstructed_noise[i*5 + j][:,:,0])
	# plt.savefig("Noise.png")
	# plt.clf()	

	# _, ax = plt.subplots(2, 5, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(5):
	# 		ax[i,j].imshow(reconstructed_exps[i*5 + j][:,:,0])
	# plt.savefig("Experiments.png")
	# plt.clf()

	# _, ax = plt.subplots(2, 3, sharey=True, sharex=True)
	# for i in range(2):
	# 	for j in range(3):
	# 		ax[i,j].imshow(reconstructed_styles[i*3 + j][:,:,0])
	# plt.savefig("Re_Styles.png")
	# plt.clf()
