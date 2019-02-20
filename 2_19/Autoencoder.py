from pprint import pprint as pp

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.models import Model, model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape

NUM_EPOCHS		= 5
BATCH_SIZE		= 70

def getData():
	(x_train, _), (x_test, _) = mnist.load_data()	#	Don't care about the y values

	#	Scale to [0,1]
	x_train = x_train.astype('float32')/float(x_train.max())
	x_test = x_test.astype('float32')/float(x_test.max())

	return (x_train, x_test)

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

	train, test = getData()
	train = train.reshape((*train.shape, 1))[:10000]
	test = test.reshape((*test.shape, 1))
	if TRAIN:
		encoder, decoder = getModel(train, test)
		saveModels((encoder, decoder))
	else:
		encoder, decoder = loadModels(('encoder', 'decoder'))

	numInterpolations = 3
	interpolationSteps = 5
	for i in range(0, 2*numInterpolations, 2):
		_, ax = plt.subplots(2,2+interpolationSteps, sharey=True, sharex=True)
		ax[0,0].imshow(test[i,:,:,0])
		ax[0,-1].imshow(test[i+1,:,:,0])

		latent_space_vec = encoder.predict(test[i:i+2,:,:])
		alpha = np.linspace(0,1,2+interpolationSteps)
		for a in alpha:
			interpolation = a*latent_space_vec[1] + (1-a)*latent_space_vec[0]
			latent_space_vec = np.vstack((latent_space_vec, interpolation))

		reconstruction = decoder.predict(latent_space_vec)

		ax[1,0].imshow(reconstruction[0,:,:,0])
		ax[1,-1].imshow(reconstruction[1,:,:,0])
		for i in range(1, interpolationSteps+1):
			ax[1,i].imshow(reconstruction[i+1,:,:,0])

		plt.show()
