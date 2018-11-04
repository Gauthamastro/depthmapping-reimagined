import random
from keras.layers import Conv2D,SeparableConv2D,Conv2DTranspose,UpSampling2D,Dense,Dropout,SpatialDropout2D,Concatenate,Add

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
class drop_layers :

	def DO(self,rate):
		self.layer = Dropout(rate,seed =seed)
		return self.layer

	def SDO(self,rate,data_format):
		self.layer =  SpatialDropout2D(rate, data_format=data_format)
		return self.layer

class dense_layers:

	def D(self,units,activation):
		self.layer = Dense(units, activation=activation)
		return self.layer

class merge_layers:
	def ADD(self,input_list):
		self.layer = Add()(input_list)
		return self.layer

class activations:
	def LeakyReLU_act(self,layer,alpha=0.3):
		self.layer = LeakyReLU(alpha=alpha)(layer)
		return self.layer

	def PReLU_act(self,layer):
		self.layer = PReLU()(layer)
		return self.layer

	def ELU_act(self,layer,alpha=1.0):
		self.layer = ELU(alpha)(layer)
		return self.layer

	def ThresoldedReLU_act(self,layer,theta=1.0):
		self.layer = ThresoldedReLU(theta)(layer)
		return self.layer

	def relu_act(self,layer, negative_slope=0.0, threshold=0.0):
		self.layer = ReLU(max_value,negative_slope,threshold)(layer)
		return self.layer

class pooling_layers:

	def MaxPooling2D_layer(self,input_layer,pool_size=(2, 2), strides=None, padding='valid', data_format=None):
		self.layer = MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer

	def AveragePooling2D_layer(self,input_layer,pool_size=(2, 2), strides=None, padding='valid', data_format=None):
		self.layer = AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer
	
	def GloblalMaxPooling2D_layer(self,input_layer,pool_size=(2, 2), strides=None, padding='valid', data_format=None):
		self.layer = GlobalMaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer
	
	def GlobalAveragePooling2D_layer(self,input_layer,pool_size=(2, 2), strides=None, padding='valid', data_format=None):
		self.layer = GlobalAveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(input_layer)
		return self.layer

class normalization:

	def BatchNormalization_layer(axis,momentum,epsilon):
		self.layer = keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001) #axis is done by assuming channels last
		return self.layer
	

class GA:
	def __init__(self,max_conv_layers,max_convT_layers,max_skip_connections,input_shape):
		#Layers available for genetic alogorithm to make
		self.special_conv_layers = ['Conv2D','SeparableConv2D']
		self.special_convT_layers =['Conv2DTranspose','UpSampling2D']
		self.drop_layers = ['Dropout','SpatialDropout2D']#No 1
		self.merge_layers = ['Add']
		self.activations = ['softmax','relu','LeakyReLU','PReLU','ELU','ThresoldedReLU']
		self.pooling_layers = ['MaxPooling2D','AveragePooling2D','GlobalMaxPooling2D','GlobalAveragePooling2D']#No 2
		self.normalization = ['BatchNormalization'] #No 3
		self.dense_layer = ['Dense']

		self.max_conv_layers=max_conv_layers
		self.max_convT_layers=max_convT_layers
		self.max_skip_connections=max_skip_connections
		self.input_shape = input_shape	

		#Params that genetic algorithm will use to search!
		self.output_filter_depth_list = [2,4,16,32,64,128,256,512,1024] # assumeing channels last
		self.kernel_size_list = [1,3,5,7,9,11]
		self.strides_list = [1,2,3,4,5]
		self.padding_list = ['same'] #['same','valid']
		self.data_format_list = ['channels_last'] #['channels_last','channels_first']
		self.momentum_list = [0.9,0.99,0.999]
		self.epsilon_list = [0.001,0.01,0.1]
		self.pool_size_list = [1,2,3]
		self.units = [2,4,16,32,64,128,256,512,1024,2048]
		self.alpha_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
		self.droputout_rate_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]



	def pickone(self,list): #helper function
		return list[random.randint(0,(len(list)-1))]


	def init_genome(self):


		self.genome = []

		for i in range(self.max_conv_layers):
			self.a = ['special_conv_layers']
			if random.randint(0,1):
				self.a.append('BatchNormalization_layer')
			if random.randint(0,1):
				self.a.append('pooling_layers')
			if random.randint(0,1):
				self.a.append('drop_layers')

			self.genome.append(self.a)

		for j in range(self.max_convT_layers):
			self.a = ['special_convT_layers']
			if random.randint(0,1):
				self.a.append('BatchNormalization')
			if random.randint(0,1):
				self.a.append('pooling_layers')
			if random.randint(0,1):
				self.a.append('drop_layers')

			self.genome.append(self.a)

		random.shuffle(self.genome)
		self.genome.insert(0,['input_layer'])
		self.genome.append(['dense_layer'])

		for i in self.genome:# for debugging only
			print(i) # for debugging only
		print('\n')
		print('\n')


		for i in range(len(self.genome)):
			for j in range(len(self.genome[i])):
				if self.genome[i][j] == 'special_conv_layers':
					if random.randint(0,1):
						self.gene = {'layer':'Conv2D',
										'filters':self.pickone(self.output_filter_depth_list),
										'kernel_size':self.pickone(self.kernel_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										'activation':self.pickone(self.activations)}
						self.genome[i][j] = self.gene
					
					else:
						self.gene = {'layer':'SeparableConv2D',
										'filters':self.pickone(self.output_filter_depth_list),
										'kernel_size':self.pickone(self.kernel_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										'activation':self.pickone(self.activations)}
						self.genome[i][j] = self.gene
					
				if self.genome[i][j] == 'special_convT_layers':
					if random.randint(0,1):
						self.gene = {'layer':'Conv2DTranspose',
										'filters':self.pickone(self.output_filter_depth_list),
										'kernel_size':self.pickone(self.kernel_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										'activation':self.pickone(self.activations)}
						self.genome[i][j] = self.gene
					
					else:
						self.gene = {'layer':'UpSampling2D',
										'filters':self.pickone(self.output_filter_depth_list),
										'kernel_size':self.pickone(self.kernel_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										'activation':self.pickone(self.activations)}
						self.genome[i][j] = self.gene
					
				if self.genome[i][j] == 'pooling_layers':
					k  = random.randint(1,4)
					if k == 1:
						self.gene = {'layer':'MaxPooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene
					
					if k == 2:
						self.gene = {'layer':'GlobalMaxPooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene
					
					if k == 3:
						self.gene = {'layer':'AveragePooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene
					
					if k == 4:
						self.gene = {'layer':'GlobalAveragePooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene
					
				if self.genome[i][j] == 'drop_layers':
					if random.randint(0,1):
						self.gene = {'layer':'Dropout',
										'rate':self.pickone(self.droputout_rate_list),
										'data_format':self.pickone(self.data_format_list)}
						self.genome[i][j] = self.gene
					
					else:
						self.gene = {'layer':'SpatialDropout2D',
										'rate':self.pickone(self.droputout_rate_list),
										'data_format':self.pickone(self.data_format_list)}
						self.genome[i][j] = self.gene

		return self.genome 





	def decode_genome(self,genome):
		#genome is a list of layers and their params
		for i in self.genome:# for debugging only!
			for j in i:
				print(j)




a = GA(20,15,0,(100,100,3),)
a.decode_genome(a.init_genome())