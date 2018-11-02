import tensorflow as tf 
import numpy as np


# use xavier initialliser and conv2d_transpose
observationsx=tf.placeholder(shape=[None,img_width,img_height,channels],dtype=tf.float32,name='observations')

initializer=tf.contrib.layers.xavier_initializer()
##nn for the enc dec

#encoder

ew1= tf.Variable(tf.initializer([5,5,channels,64]))
ew2= tf.Variable(tf.random_normal([5,5,64,64], seed=seed))
ew3= tf.Variable(tf.random_normal([img_height*img_width*64,1024], seed=seed))
ew4= tf.Variable(tf.random_normal([1024,512], seed=seed))
ew5= tf.Variable(tf.random_normal([512,216], seed=seed))


eb1= tf.Variable(tf.random_normal([64], seed=seed))
eb2= tf.Variable(tf.random_normal([64], seed=seed))
eb3= tf.Variable(tf.random_normal([1024], seed=seed))
eb4= tf.Variable(tf.random_normal([512], seed=seed))
eb5= tf.Variable(tf.random_normal([216], seed=seed))





el1 = tf.add(tf.nn.conv2d(observationsx,ew1,strides=[1,1,1,1],padding='SAME'),eb1)
el1 = tf.nn.relu(el1)

el2 = tf.add(tf.nn.conv2d(el1, ew2,strides=[1,1,1,1],padding='SAME'),eb2)
el2 = tf.nn.relu(el2)


flat= tf.reshape(el2,[-1,img_height*img_width*64])
el3 = tf.add(tf.matmul(flat, ew3),eb3)
el3 = tf.nn.relu(el3)

el4 = tf.add(tf.matmul(el3, ew4),eb4)
el4 = tf.nn.relu(el4)

#output
elout = tf.add(tf.matmul(el4, ew5),eb5)
elout = tf.nn.relu(elout)

encoder_output=tf.multinomial(logits=lout,num_samples=1)

#decoder with conv2dtrans without skip connections
"""
w1= tf.Variable(tf.initializer([5,5,channels,64]))
w2= tf.Variable(tf.random_normal([5,5,64,64], seed=seed))
w3= tf.Variable(tf.random_normal([img_height*img_width*64,1024], seed=seed))
w4= tf.Variable(tf.random_normal([1024,512], seed=seed))
w5= tf.Variable(tf.random_normal([512,216], seed=seed))


b1= tf.Variable(tf.random_normal([64], seed=seed))
b2= tf.Variable(tf.random_normal([64], seed=seed))
b3= tf.Variable(tf.random_normal([1024], seed=seed))
b4= tf.Variable(tf.random_normal([512], seed=seed))
b5= tf.Variable(tf.random_normal([216], seed=seed))

dl1=tf.add(tf.matmul(elout,dw1),db1)
dl1=tf.nn.relu(dl1)
"""

with tf.Session() as sess:
	tf.initialize_all_variables().run()
	(img_height,img_width,channels)=np.shape(input_img)
	#input_img =input_img.ravel()
	#inpt=inpt.reshape(img_height,img_width,channels)
	"""
	sess.run(valid_pad)

	sess.run(same_pad)
	print(valid_pad.get_shape() )
	print(same_pad.get_shape())
	a,b,c,d=valid_pad.get_shape()
	print a+b+c+d
	"""
	action=sess.run(sample_op,feed_dict={observationsx:[input_img]})
	