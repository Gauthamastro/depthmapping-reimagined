from keras.models import Sequential, Model 
from keras.layers import Input,SeparableConv2D,MaxPooling2D,BatchNormalization

#Build the network
def conv_block(input_layer):

	conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
	conv1 = BatchNormalization()(conv1)
	conv1 = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)


	return conv5


img_width,img_height = 1024,1024

img_input = Input(shape=(img_width, img_height, 3))

model = Model(input = img_input, output = conv_block(img_input))

model.summary()


#This conv5 feature map is feed as the label to encoder


layer_dict = dict([(layer.name, layer) for layer in model.layers])
print([layer.name for layer in model.layers])