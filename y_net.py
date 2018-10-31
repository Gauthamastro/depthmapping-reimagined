from conv_block import conv_block
from decoder import decoder_block
from encoder import encoder_block
from keras.layers import Input
from keras.models import Model
from keras.losses import kullback_leibler_divergence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

img_width,img_height = 440,800

img_input = Input(shape=(img_width, img_height, 3))#RGB image
dis_input = Input(shape=(img_width,img_height,1))#Disparity image
feature_map = Input(shape=(64,64,512))#This will be shape of feature map!

#Location of data
train_rgb_folder = 'data/train/rgb/'
test_rgb_folder = 'data/test/rgb/'
val__rgb_folder = 'data/val/rgb'

train_depth_folder = 'data/train/depth/'
test_depth_folder = 'data/test/depth/'
val__depth_folder = 'data/val/depth'

import os
import cv2

a = os.listdir(train_rgb_folder)
a.sort()
rgb = []
for i in a:
	d = cv2.imread(train_rgb_folder+i)
	d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
	rgb.append(d)
print(len(a))
depth = []#np.load('depth.npy')
for i in a:
	d = cv2.imread(train_rgb_folder+i)
	d = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
	depth.append(d)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(rgb, depth, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

#conv_model = Model(input = img_input, output = conv_block(img_input))
#conv_model.summary()

#encoder_model = Model(input= dis_input,output=encoder_block(dis_input))
#encoder_model.summary()

#decoder_model = Model(input = feature_map, output = decoder_block(feature_map))
#decoder_model.summary()



#now rectify the syntax and various other errors
#figure out a way to get the encoder output from the auto_enc_dec_model so as to act as label for the conv_model
# then compile the models with suitable loss and regularisation
#and then all we need to preprocess the images and disparity images
#do some intial training of the auto_enc_dec_model 
# then iteratively train the conv_model taking encoder output as label, then the auto_enc_dec_model to give better encoding
# can also exchange weights if stable training is needed
#also need to change the structure of decoder to take a image input as a reference for better segmnetation


def training_enc_dec(epochs,batch_size,disparity_map_val,disparity_map):
	depth_input_layer= Input((img_width,img_height,1))
	center_feature_vector = encoder_block(depth_input_layer)
	final_disparity_map = decoder_block(center_feature_vector)

	auto_enc_dec_model = Model(input= depth_input_layer,output=final_disparity_map)
	auto_enc_dec_model.compile(loss=kullback_leibler_divergence,optimizer='adam',metrics=['accuracy'])
	#auto_enc_dec_model.summary()
	# checkpoint
	filepath="weights-improvement-enc_dec-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#print(disparity_map.shape)
	#print(disparity_map_val.shape)
	auto_enc_dec_model.fit(disparity_map, disparity_map,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(disparity_map_val, disparity_map_val),
        callbacks=callbacks_list)


def conv_enc(batch_size,epochs,val_x,val_y,img_input,disparity_map):
	disparity_map= Input((img_width,img_height,1))
	center_feature_vector = encoder_block(disparity_map)
	encoder_model = Model(input=disparity_map,output=center_feature_vector)
	encoder_model.load_weights('path to weights file',by_name=True)
	feature_maps = encoder_model.predict(disparity_map)
	img_input_layer= Input((img_width,img_height,3))
	conv_model_layer = conv_block(img_input_layer)
	#Disparity_map and img_put should have same number of samples
	conv_model = Model(input=img_input,output=conv_model_layer)
	conv_model.compile(loss=kullback_leibler_divergence,optimizer='adam',metrics=['accuracy'])
	#Checkpoints
	filepath="weights-improvement-enc_conv-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#Training
	conv_model.fit(img_input,feature_maps,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1,
		callbacks=callbacks_list,
		validation_data=(val_x,val_y))



training_enc_dec(32,10,np.array([y_val]),np.array([y_train]))



