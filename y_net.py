import conv_block
import decoder




img_width,img_height = 1024,1024

img_input = Input(shape=(img_width, img_height, 3))

conv_model = Model(input = img_input, output = conv_block(img_input))

conv_model.summary()


auto_enc_dec_model = Model(input = img_input, output = decoder_block(img_input))

auto_enc_dec_model.summary()



#now rectify the syntax and various other errors
#figure out a way to get the encoder output from the auto_enc_dec_model so as to act as label for the conv_model
# then compile the models with suitable loss and regularisation
#and then all we need to preprocess the images and disparity images
#do some intial training of the auto_enc_dec_model 
# then iteratively train the conv_model taking encoder output as label, then the auto_enc_dec_model to give better encoding
# can also exchange weights if stable training is needed
#also need to change the structure of decoder to take a image input as a reference for better segmnetation
