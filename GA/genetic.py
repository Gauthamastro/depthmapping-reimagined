import random
from keras.layers import Conv2D,Separable2D,Conv2DTranspose,UpSampling2D,Dense,Dropout,SpatialDropout2D,Concatenate,Add

seed =34
random.seed(34)

#We give optimal parameters in the init fn of these classes !
#these will be accessed on execution for mutation and crossover!


class conv_layers:
	def C2D(self,filters,kernel_size,strides,padding,data_format,activation):
		self.layer = Conv2D(filters,kernel_size, strides=strides,
							padding=padding, data_format=data_format, activation=activation)
		return self.layer #self.layer.output_shape

	def S2D(self,filters,kernel_size,strides,padding,data_format,activation):
		self.layer = SeparableConv2D(filters,kernel_size, strides=strides,
									 padding=padding, data_format=data_format, activation=activation)
		return self.layer


class convT_layers:
	def C2DT(self,filters,kernel_size,strides,padding,data_format,activation):
		self.layer=keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, 
												padding=padding, output_padding=None, data_format=data_format,activation=activation)
		return self.layer

	def US2D(self,filters,kernel_size,strides,padding,data_format,activation):
		self.layer=UpSampling2D(filters, kernel_size, strides=strides,
												 padding=padding, output_padding=None, data_format=data_format,activation=activation)
		return self.layer
class core_layers :
	def D(self,activation):
		self.layer = Dense(units, activation=activation)
		return self.layer

	def DO(self,rate):
		self.layer = Dropout(rate,seed =seed)
		return self.layer

	def SDO(self,rate,data_format):
		self.layer =  SpatialDropout2D(rate, data_format=data_format)
		return self.layer

class merge_layers:
	def CON(self,input_list):
		self.layer = Concatenate(axis=-1)(input_list)
		return self.layer

	def ADD(self,input_list):
		self.layer = Add()(input_list)
		return self.layer
class activations:
	def softmax_act(self,layer):
		self.layer = Softmax(axis=-1)(layer)
		return self.layer

	def LeakyReLU_act(self,layer,alpha=0.3):
		self.layer = LeakyReLU(alpha=0.3)(layer)
		return self.layer

	def PReLU_act(self,layer):
		self.layer = PReLU()(layer)
		return self.layer

	def ELU_act(self,layer,alpha=1.0):
		self.layer = ELU()(layer)
		return self.layer

	def ThresoldedReLU_act(self,layer,theta=1.0):
		self.layer = ThresoldedReLU(theta)(layer)
		return self.layer

	def relu_act(self,layer,max_value=None, negative_slope=0.0, threshold=0.0):
		self.layer = ReLU(max_value,negative_slope,threshold)(layer)
		return self.layer
class pooling_layers:
	def MaxPooling2D_layer(self,pool_size=(2, 2), strides=None, padding='valid', data_format=None,input_layer):
		self.layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer

	def AveragePooling2D_layer(self,pool_size=(2, 2), strides=None, padding='valid', data_format=None,input_layer):
		self.layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer
	
	def GloblalMaxPooling2D_layer(self,pool_size=(2, 2), strides=None, padding='valid', data_format=None,input_layer):
		self.layer = GlobalMaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer
	
	def GlobalAveragePooling2D_layer(self,pool_size=(2, 2), strides=None, padding='valid', data_format=None,input_layer):
		self.layer = GlobalAveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer

class normalization:
	def BatchNormalization_layer():
		self.layer = 
		return self.layer
	

class GA:
	def __init__(self,max_conv_layers,max_convT_layers,max_skip_connections,input_shape,activations,normalization):
		self.special_conv_layers = ['Conv2D','Separable2D']
		self.special_convT_layers =['Conv2DTranspose','UpSampling2D']
		self.core_layers = ['Dense','Dropout','SpatialDropout2D']
		self.merge_layers = ['Concatenate','Add']
		self.activations = ['softmax','relu','LeakyReLU','PReLU','ELU','ThresoldedReLU']
		self.pooling_layers = ['MaxPooling2D','AveragePooling2D','GlobalMaxPooling2D','GlobalAveragePooling2D']
		self.normalization = ['BatchNormalization']

		self.max_conv_layers=max_conv_layers
		self.max_convT_layers=max_convT_layers
		self.max_skip_connections=max_skip_connections
		self.input_shape = input_shape
		self.activations = activations
		self.normalization= normalization





	def init_genome(self):


		gene = []






	def decode_genome(self,genome):
		#genome is a list of layers and their params
		for gene in genome:
			if gene[0] in self.special_conv_layers:
				pass
			elif gene[0] in self.special_convT_layers:
				pass
			elif gene[0] in self.special_layers:
				pass
			elif gene[0] in self.core_layers:
				pass
			elif gene[0] in self.merge_layers:
				pass
			elif gene[0] in self.activations:
				pass
			elif gene[0] in self.pooling_layers:
				pass
			elif gene[0] in self.normalization:
				pass