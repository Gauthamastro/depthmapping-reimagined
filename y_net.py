from conv_block import conv_block
from decoder import decoder_block
from encoder import encoder_block
from keras.layers import Input
from keras.models import Model
from keras.losses import kullback_leibler_divergence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint



img_width,img_height = 1024,1024

img_input = Input(shape=(img_width, img_height, 3))#RGB image
dis_input = Input(shape=(img_width,img_height,1))#Disparity image
feature_map = Input(shape=(64,64,512))#This will be shape of feature map!
disparity_map_val = [] #Define a validation set of disparity maps for auto encoder decoder network!


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


def training_enc_dec(disparity_map,epochs=10,batch_size=1,disparity_map_val=None):
	center_feature_vector = encoder_block(disparity_map)
	final_disparity_map = decoder_block(center_feature_vector)

	auto_enc_dec_model = Model(input= disparity_map,output=final_disparity_map)
	auto_enc_dec_model.compile(loss=kullback_leibler_divergence,optimizer='adam',metrics=['accuracy'])
	auto_enc_dec_model.summary()
	# checkpoint
	filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	auto_enc_dec_model.fit(disparity_map, disparity_map,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(disparity_map_val, disparity_map_val),
        callbacks=callbacks_list)

