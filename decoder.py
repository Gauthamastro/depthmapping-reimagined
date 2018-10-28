from keras.models import Sequential, Model 
from keras.layers import Input,SeparableConv2D,MaxPooling2D,BatchNormalization,Conv2DTranspose,Dropout,Conv2D
from conv_block import *

def decoder_block(feature_layer):


	up1 = Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(feature_layer)
	conv1 = Conv2D(256,(3,3),activation='relu',padding='same')(up1)
	conv1 = BatchNormalization()(conv1)
	conv1 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Dropout(rate=0.4)(conv1)


	up2 = Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(conv1)
	conv2 = Conv2D(128,(3,3),activation='relu',padding='same')(up2)
	conv2 = BatchNormalization()(conv2)
	conv2 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Dropout(rate=0.4)(conv2)


	up3 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(conv2)
	conv3 = Conv2D(64,(3,3),activation='relu',padding='same')(up3)
	conv3 = BatchNormalization()(conv3)
	conv3 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Dropout(rate=0.4)(conv3)


	up4 = Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(conv3)
	conv4 = Conv2D(64,(3,3),activation='relu',padding='same')(up4)
	conv4 = BatchNormalization()(conv4)
	conv4 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Dropout(rate=0.4)(conv4)

	up5 = Conv2DTranspose(1,(1,1),strides=(1,1),padding='same')(conv4)
	return up5

'''
input_layer = Input(shape=(64, 64, 512))

model = Model(input = input_layer, output = decoder_block(input_layer))

model.summary()
'''
