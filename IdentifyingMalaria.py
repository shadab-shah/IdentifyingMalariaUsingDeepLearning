'''
Identifying Malaria using Deep Learning
'''



import resnet
from imutils import paths
import random
import shutil
import os
import matplotlib
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

#Intializing the dataset
ORIG_INPUT_DATASET = "..\Maleria Detection\cell_images"
BASE_PATH = "..\Maleria Detection"
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1


imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

i = int(len(imagePaths) * config.TRAIN_SPLIT) # number of image in training folder
trainPaths = imagePaths[:i] # first path to the ith part
testPaths = imagePaths[i:] # from the ith location to the last location

i = int(len(trainPaths) * config.VAL_SPLIT) # from traning folder divide it into validation
valPaths = trainPaths[:i] # first path to the ith part
trainPaths = trainPaths[i:] # from the ith location to the last location

# define the datasets that we'll be building
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# the dataset variable created above is been used in for loop to loop through
    # the data populated in the dataset variable.
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image along with its
		# corresponding class label
		#print(inputPath.split(os.path.sep)[-1])
		#print(inputPath.split(os.path.sep)[-2])
		
		filename = inputPath.split(os.path.sep)[-1]
		label = inputPath.split(os.path.sep)[-2]

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])
		# within the testing folder we are creating 2 folders by the name of Uninfected 
		# and Parasitized.
		#print(labelPath)

		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)

		# construct the path to the destination image and then copy
		# the image itself
		p = os.path.sep.join([labelPath, filename])
		shutil.copy2(inputPath, p)
        
        
# set the matplotlib backend so figures can be saved in the background

matplotlib.use("Agg")



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define the total number of epochs to train for along with the
# initial learning rate and batch size
NUM_EPOCHS = 20
INIT_LR = 1e-1
BS = 32

totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

class ResNet:
	@staticmethod
	def residual_module(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
		# the shortcut branch of the ResNet module should be
		# initialize as the input (identity) data
		shortcut = data

		# the first block of the ResNet module are the 1x1 CONVs
		bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(data)
		act1 = Activation("relu")(bn1)
		conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act1)

		# the second block of the ResNet module are the 3x3 CONVs
		bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv1)
		act2 = Activation("relu")(bn2)
		conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
			padding="same", use_bias=False,
			kernel_regularizer=l2(reg))(act2)

		# the third block of the ResNet module is another set of 1x1
		# CONVs
		bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
			momentum=bnMom)(conv2)
		act3 = Activation("relu")(bn3)
		conv3 = Conv2D(K, (1, 1), use_bias=False,
			kernel_regularizer=l2(reg))(act3)

		# if we are to reduce the spatial size, apply a CONV layer to
		# the shortcut
		if red:
			shortcut = Conv2D(K, (1, 1), strides=stride,
				use_bias=False, kernel_regularizer=l2(reg))(act1)

		# add together the shortcut and the final CONV
		x = add([conv3, shortcut])

		# return the addition as the output of the ResNet module
		return x
    

def build(width, height, depth, classes, stages, filters,
	reg=0.0001, bnEps=2e-5, bnMom=0.9):
	# initialize the input shape to be "channels last" and the
	# channels dimension itself
	inputShape = (height, width, depth)
	chanDim = -1

	# if we are using "channels first", update the input shape
	# and channels dimension
	if K.image_data_format() == "channels_first":
		inputShape = (depth, height, width)
		chanDim = 1

	# set the input and apply BN
	inputs = Input(shape=inputShape)
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(inputs)

	# apply CONV => BN => ACT => POOL to reduce spatial size
	x = Conv2D(filters[0], (5, 5), use_bias=False,
		padding="same", kernel_regularizer=l2(reg))(x)
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(x)
	x = Activation("relu")(x)
	x = ZeroPadding2D((1, 1))(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# loop over the number of stages
	for i in range(0, len(stages)):
		# initialize the stride, then apply a residual module
		# used to reduce the spatial size of the input volume
		stride = (1, 1) if i == 0 else (2, 2)
		x = ResNet.residual_module(x, filters[i + 1], stride,
			chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

		# loop over the number of layers in the stage
		for j in range(0, stages[i] - 1):
			# apply a ResNet module
			x = ResNet.residual_module(x, filters[i + 1],
				(1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

	# apply BN => ACT => POOL
	x = BatchNormalization(axis=chanDim, epsilon=bnEps,
		momentum=bnMom)(x)
	x = Activation("relu")(x)
	x = AveragePooling2D((8, 8))(x)

	# softmax classifier
	x = Flatten()(x)
	x = Dense(classes, kernel_regularizer=l2(reg))(x)
	x = Activation("softmax")(x)

	# create the model
	model = Model(inputs, x, name="resnet")

	# return the constructed network architecture
	return model

# initialize our ResNet model and compile it
#model = ResNet50(weights='imagenet')
model = build(64, 64, 3, 2, (3, 4, 6),	(64, 128, 256, 512), reg=0.0005)

opt = SGD(lr=INIT_LR, momentum=0.9)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = NUM_EPOCHS
	baseLR = INIT_LR
	power = 1.0

	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

	# return the new learning rate
	return alpha

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]

H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=callbacks)
    
with open('model.pkl', 'wb') as f:  
    pickle.dump([model], f)    
	
testGen.reset()
predIdxs = model.predict_generator(testGen,	steps=(totalTest // BS) + 1)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))    
