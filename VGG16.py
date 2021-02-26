import tensorflow as tf
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def makeModel(activation):
	model = tf.keras.models.Sequential([
		layers.Input(shape = (img_height, img_width, 3)),
		# 1st conv
		tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'),
		# 2nd conv
		tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'),
		# 3rd conv
		tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'),
		# 4th conv
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'),
		# 5th Conv
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding = 'same', activation = activation),
		tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, padding = 'same'),
		# Fully connected layers
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(units = 4096, activation = activation),
		tf.keras.layers.Dense(units = 4096, activation = activation),
		tf.keras.layers.Dense(2, activation = 'softmax')
	])
	return model

epochs = 10

root = "A:/Master/DeepLearning/Data/"
dataDir = root + "SortedData/"

batch_size = 32
img_height = 218
img_width = 178

tf.keras.backend.clear_session()

for x in range(0,10):

	print("--------------Model1---------------")
	print("---------------relu----------------")
	print("---------------SGD-----------------")
	
	seed = int(time.time())
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	model = makeModel("relu")
	
	#model.summary()
	
	model.compile(
		optimizer=tf.keras.optimizers.SGD(),
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)
	
	log_dir = "logs/fit/VGG_relu_SGD_" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	model.fit(training, validation_data=testing, epochs=epochs, callbacks=[tensorboard_callback])

	tf.keras.backend.clear_session()

	#--------------------------------------------------------------------------------

	print("--------------Model2---------------")
	print("------------leaky relu-------------")
	print("---------------SGD-----------------")
	
	seed = int(time.time())
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	model = makeModel(tf.keras.layers.LeakyReLU(alpha=0.05))
	
	#model.summary()
	
	model.compile(
		optimizer=tf.keras.optimizers.SGD(),
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	log_dir = "logs/fit/VGG_leakyRelu_SGD_" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	model.fit(training, validation_data=testing, epochs=epochs, callbacks=[tensorboard_callback])

	tf.keras.backend.clear_session()

	#--------------------------------------------------------------------------------

	print("--------------Model3---------------")
	print("---------------relu----------------")
	print("-----------SGD nesterov------------")
	
	seed = int(time.time())
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	model = makeModel("relu")
	
	#model.summary()
	
	model.compile(
		optimizer=tf.keras.optimizers.SGD(momentum=0.5, nesterov=True),
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	log_dir = "logs/fit/VGG_relu_nestrov_" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	model.fit(training, validation_data=testing, epochs=epochs, callbacks=[tensorboard_callback])

	tf.keras.backend.clear_session()

	#--------------------------------------------------------------------------------

	print("--------------Model4---------------")
	print("------------leaky relu-------------")
	print("-----------SGD nesterov------------")
	
	seed = int(time.time())
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(img_height, img_width),
		batch_size=batch_size)
	
	model = makeModel(tf.keras.layers.LeakyReLU(alpha=0.05))
	
	#model.summary()
	
	model.compile(
		optimizer=tf.keras.optimizers.SGD(momentum=0.5, nesterov=True),
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	log_dir = "logs/fit/VGG_leakyRelu_nestrov_" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	model.fit(training, validation_data=testing, epochs=epochs, callbacks=[tensorboard_callback])

	tf.keras.backend.clear_session()

	#--------------------------------------------------------------------------------
