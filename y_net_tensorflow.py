import tensorflow as tf 
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
from PIL import Image
(img_height,img_width,channels)=(100,100, 3)
# use xavier initialliser and conv2d_transpose
#batch_size=tf.placeholder(shape=[None],dtype=tf.uint8,name='batch_size')
batch_size=1
learning_rate=tf.placeholder(shape=[None],dtype=tf.float32,name='learning_rate')
observationsx=tf.placeholder(shape=[batch_size,img_width,img_height,channels],dtype=tf.float32,name='observations')

initializer=tf.contrib.layers.xavier_initializer()
##nn for the enc dec

#encoder
seed=32
ew1= tf.Variable(initializer([5,5,channels,64]))#[h,w,inputchannels,outputchannels]
ew2= tf.Variable(tf.random_normal([5,5,64,64], seed=seed))
ew3= tf.Variable(tf.random_normal([img_height*img_width*64,1024], seed=seed)) 

'''This is the line above that is making the problem ! it cant allocate such a large tensor [640000,1024]
	u need to change the dimensions! 

'''
ew4= tf.Variable(tf.random_normal([1024,512], seed=seed))
ew5= tf.Variable(tf.random_normal([512,216], seed=seed))


eb1= tf.Variable(tf.random_normal([64], seed=seed))
eb2= tf.Variable(tf.random_normal([64], seed=seed))
eb3= tf.Variable(tf.random_normal([1024], seed=seed))
eb4= tf.Variable(tf.random_normal([512], seed=seed))
eb5= tf.Variable(tf.random_normal([216], seed=seed))





el1 = tf.add(tf.nn.conv2d(observationsx,ew1,strides=[batch_size,1,1,1],padding='SAME'),eb1)
el1 = tf.nn.relu(el1)

el2 = tf.add(tf.nn.conv2d(el1, ew2,strides=[batch_size,1,1,1],padding='SAME'),eb2)
el2 = tf.nn.relu(el2)


flat= tf.reshape(el2,[batch_size,img_height*img_width*64])
el3 = tf.add(tf.matmul(flat, ew3),eb3)
el3 = tf.nn.relu(el3)

el4 = tf.add(tf.matmul(el3, ew4),eb4)
el4 = tf.nn.relu(el4)

#output
elout = tf.add(tf.matmul(el4, ew5),eb5)
elout = tf.nn.relu(elout)

#encoder_output=tf.multinomial(logits=lout,num_samples=1)

#decoder with conv2dtrans without skip connections

dw1= tf.Variable(initializer([5,5,1,64]))#[h,w,opchannels,inchannels]
dw2= tf.Variable(tf.random_normal([5,5,64,64], seed=seed))
dw3= tf.Variable(tf.random_normal([1024,img_width*img_height*64], seed=seed))
dw4= tf.Variable(tf.random_normal([512,1024], seed=seed))
dw5= tf.Variable(tf.random_normal([216,512], seed=seed))


db1= tf.Variable(tf.random_normal([1], seed=seed))
db2= tf.Variable(tf.random_normal([64], seed=seed))
db3= tf.Variable(tf.random_normal([img_width*img_height*64], seed=seed))
db4= tf.Variable(tf.random_normal([1024], seed=seed))
db5= tf.Variable(tf.random_normal([512], seed=seed))

dl1=tf.add(tf.matmul(elout,dw5),db5)
dl1=tf.nn.relu(dl1)

dl2=tf.add(tf.matmul(dl1,dw4),db4)
dl2=tf.nn.relu(dl2)


dl3=tf.add(tf.matmul(dl2,dw3),db3)
dl3=tf.nn.relu(dl3)

un_flat=tf.reshape(dl3,[-1,img_height,img_width,64])

dl4=tf.add(tf.nn.conv2d_transpose(un_flat,dw2,output_shape=[batch_size,img_height,img_width,64],strides=[batch_size,1,1,1],padding='SAME'),db2)
dl4=tf.nn.relu(dl4)

dl5=tf.add(tf.nn.conv2d_transpose(dl4,dw1,output_shape=[batch_size,img_height,img_width,1],strides=[batch_size,1,1,1],padding='SAME'),db1)

output_image=tf.nn.relu(dl5)   #find a better activation here

#loss=



with tf.Session() as sess:
	tf.initialize_all_variables().run()
	input_img=cv2.imread("/home/akash/Desktop/my_car/real_data/f(0).jpg")
	input_img= cv2.resize(input_img,(100,100))
	#cv2.imshow('input image',input_img)
	print (np.shape(input_img))
	i = Image.fromarray(input_img)
	i.show()
	output=sess.run(output_image,feed_dict={observationsx:[input_img]})
	#print output
	print (np.shape(output[0]))
	output=output.ravel()
	output=np.reshape(output,(100,100))
	i = Image.fromarray(output)
	i.show()
	#output=sess.run(output_image,feed_dict={observationsx:[input_img]})
"""
input_img=cv2.imread("/home/akash/Desktop/my_car/real_data/f(0).jpg")
print np.shape(input_img)#(480, 640, 3)
input_img= cv2.resize(input_img,(100,100))
print np.shape(input_img)
"""
