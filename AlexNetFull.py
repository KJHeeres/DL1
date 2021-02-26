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
		layers.experimental.preprocessing.CenterCrop(height=227, width=227, input_shape=(new_img_height, new_img_width, 3)),
		#layers.experimental.preprocessing.Rescaling(1./255),
		# 1st conv
		tf.keras.layers.Conv2D(96, (11,11),strides=(4,4), activation=activation),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPooling2D(3, strides=(2,2)),
		# 2nd conv
		tf.keras.layers.Conv2D(256, (5,5),strides=(1,1), activation=activation,padding="same"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPooling2D(3, strides=(2,2)),
		# 3rd conv
		tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation=activation,padding="same"),
		tf.keras.layers.BatchNormalization(),
		# 4th conv
		tf.keras.layers.Conv2D(384, (3,3),strides=(1,1), activation=activation,padding="same"),
		tf.keras.layers.BatchNormalization(),
		# 5th Conv
		tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation=activation,padding="same"),
		tf.keras.layers.BatchNormalization(),
		tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
		# To Flatten layer
		tf.keras.layers.Flatten(),
		# To FC layer 1
		tf.keras.layers.Dense(4096, activation=activation),
		tf.keras.layers.Dropout(0.5),
		#To FC layer 2
		tf.keras.layers.Dense(4096, activation=activation),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(2, activation='softmax')
	])
	return model

epochs = 15

root = "D:/AI/3/Deep Learning/Dataset/"
dataDir = root + "mf/"
print(dataDir)

batch_size = 32
new_img_height = int(227.0/178.0 * 218) # from 218
new_img_width = 227 # from 178

for x in range(0,5):
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+         +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+  RUN " + str(x) + "  +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+         +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	
	seed = random.randint(0, 1000000)
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	model1 = makeModel("relu")
	
	model1.summary()
	
	print("--------------Model1---------------")
	print("---------------relu----------------")
	print("---------------SGD-----------------")
	
	model1.compile(
	  optimizer=tf.keras.optimizers.SGD(),
	  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy']
	)
	
	log_dir = "logs/fit/Alex_relu_SGD" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	model1.fit(training, validation_data=testing, epochs=epochs)


	#--------------------------------------------------------------------------------

	
	seed = random.randint(0, 1000000)
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	model2 = makeModel(tf.keras.layers.LeakyReLU(alpha=0.05))
	
	#model2.summary()
	
	print("--------------Model2---------------")
	print("------------leaky relu-------------")
	print("---------------SGD-----------------")
	
	model2.compile(
	  optimizer=tf.keras.optimizers.SGD(),
	  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy']
	)
	
	log_dir = "logs/fit/Alex_leakyrelu_SGD" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	model2.fit(training, validation_data=testing, epochs=epochs)



	#--------------------------------------------------------------------------------


	
	seed = random.randint(0, 1000000)
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	model3 = makeModel("relu")
	
	#model3.summary()
	
	print("--------------Model3---------------")
	print("---------------relu----------------")
	print("-----------SGD nesterov------------")
	
	model3.compile(
	  optimizer=tf.keras.optimizers.SGD(momentum=0.2, nesterov=True),
	  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy']
	)

	log_dir = "logs/fit/Alex_relu_nesterov_" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	model3.fit(training, validation_data=testing, epochs=epochs)



	#--------------------------------------------------------------------------------


	
	seed = random.randint(0, 1000000)
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	model4 = makeModel(tf.keras.layers.LeakyReLU(alpha=0.05))
	
	#model4.summary()
	
	print("--------------Model4---------------")
	print("------------leaky relu-------------")
	print("-----------SGD nesterov------------")
	
	model4.compile(
	  optimizer=tf.keras.optimizers.SGD(momentum=0.2, nesterov=True),
	  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy']
	)

	log_dir = "logs/fit/Alex_leakyrelu_nesterov" + str(x)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	
	model4.fit(training, validation_data=testing, epochs=epochs)



	#--------------------------------------------------------------------------------


	'''
	seed = random.randint(0, 1000000)
	
	training = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="training",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)
	
	testing = tf.keras.preprocessing.image_dataset_from_directory(
		dataDir,
		validation_split=0.2,
		subset="validation",
		seed=seed,
		image_size=(new_img_height, new_img_width),
		batch_size=batch_size)

	model5 = makeModel("relu")
	
	#model4.summary()
	
	print("--------------Model5---------------")
	print("---------------relu----------------")
	print("---------------Adam----------------")
	
	model5.compile(
	  optimizer="adam",
	  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
	  metrics=['accuracy']
	)
	
	model5.fit(training, validation_data=testing, epochs=epochs)
	'''

	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+         +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+  RUN " + str(x) + "  +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+         +#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	print("-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-+#+-")
	
#model.save("sorted_limited_model")

# tutorials used:
# https://medium.com/analytics-vidhya/alexnet-tensorflow-2-1-0-d398b7c76cf
# https://learnopencv.com/understanding-alexnet/