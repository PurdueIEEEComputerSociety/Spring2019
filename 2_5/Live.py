
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pp
from keras import Sequential
from keras.layers import LSTM, Dense
from pandas import read_csv

TEST_PORTION = 0.8
NUM_EPOCHS = 50
BATCH_SIZE = 70

def getData(directory, filename):
	data = read_csv(directory + "/" + filename)

	#	Drop non-numeric data
	data.drop(labels=['date','label'], axis=1, inplace=True)

	#	Normalize to [0,1]
	values = data.values/data.values.max(axis=0)

	print(values.shape)
	pp(values[0:3,:])

	#	Append the output vector
	inputs = values[:-1]
	outputs = values[1:,3].reshape(values[1:,3].shape[0],1)
	values = np.concatenate((inputs, outputs), 1)

	#	Test/Train split
	train = values[:int(values.shape[0] * TEST_PORTION)]
	test = values[int(values.shape[0] * TEST_PORTION):]

	print((train.shape, test.shape))

	x = np.array(list(range(values.shape[0])))
	plt.plot(x, values[:,3])
	plt.show()
	plt.clf()

	x_train = x[:int(x.shape[0] * TEST_PORTION)]
	x_test = x[int(x.shape[0] * TEST_PORTION):]
	plt.plot(x_train, train[:,3])
	plt.plot(x_test, test[:,3])
	plt.show()
	plt.clf()

	return (train, test)

def getModel(train_x, train_y, test_x, test_y):
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_x.shape[1:])))
	model.add(Dense(1))
	model.compile(loss='mse', optimizer='adam')

	#	Train
	history = model.fit(train_x, train_y,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		validation_data=(test_x, test_y),
		shuffle=False)

	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()
	plt.clf()

	return model

def stockPolicy(market):
	return market[1:] > market[:-1]

if __name__ == '__main__':
	train, test = getData("data", "AAPL.csv")

	#	Reshape into format expected by LSTMS
	train_x, train_y = train[:,:-1], train[:,-1]
	test_x, test_y = test[:,:-1], test[:,-1]
	train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
	test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

	model = getModel(train_x, train_y, test_x, test_y)

	y_hat = model.predict(test_x)

	#	Perform some analytical analysis
	real = test_y[1:]
	naive = test_y[:-1]	#	Assume t(i+1) = t(i)
	predicted = y_hat[1:].reshape(y_hat[1:].shape[0])

	plt.plot(real, label='real')
	plt.plot(naive, label='naive')
	plt.plot(predicted, label='predicted')
	plt.legend()
	plt.show()
	plt.clf()

	placedBets = stockPolicy(predicted)
	optimalBets = stockPolicy(real)
	naiveBets = stockPolicy(naive)

	print(optimalBets.shape)
	print(sum(placedBets == optimalBets))
	print(sum(naiveBets == optimalBets))





