from keras.models import Sequential, Model 
from keras.layers import Input,SeparableConv2D,MaxPooling2D,BatchNormalization


def encoder_block(input_layer):

	conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',name ='enc1')(input_layer)
	conv1 = BatchNormalization(name ='enc1.1')(conv1)
	conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same',name ='enc1.2')(conv1)
	conv1 = BatchNormalization(name ='enc1.3')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2),name ='enc1.4')(conv1)

	conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',name ='enc2')(pool1)
	conv2 = BatchNormalization(name ='enc2.1')(conv2)
	conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same',name ='enc2.2')(conv2)
	conv2 = BatchNormalization(name ='enc2.3')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2),name ='enc2.4')(conv2)

	conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',name='enc3')(pool2)
	conv3 = BatchNormalization(name='enc3.1')(conv3)
	conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same',name='enc3.2')(conv3)
	conv3 = BatchNormalization(name='enc3.3')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2),name='enc3.4')(conv3)

	conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same',name='enc4')(pool3)
	conv4 = BatchNormalization(name='enc4.1')(conv4)
	conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same',name='enc4.2')(conv4)
	conv4 = BatchNormalization(name='enc4.3')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2),name='enc4.4')(conv4)

	conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same',name='enc5')(pool4)
	conv5 = BatchNormalization(name='enc5.1')(conv5)
	conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same',name='enc5.2')(conv5)
	conv5 = BatchNormalization(name='enc5.3')(conv5)


	return conv5
'''
img_width,img_height = 1024,1024

img_input = Input(shape=(img_width, img_height, 3))

model = Model(input = img_input, output = encoder_block(img_input))

model.summary()
'''