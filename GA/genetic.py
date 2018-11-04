import random,math
from keras.layers import Conv2D,SeparableConv2D,Conv2DTranspose,UpSampling2D,Dense,Dropout,SpatialDropout2D,Concatenate,Add
from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization,AveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,GlobalAveragePooling2D
import numpy as np 

seed =34
random.seed(34)

#We give optimal parameters in the init fn of these classes !
#these will be accessed on execution for mutation and crossover!

class GA:
	def __init__(self,max_conv_layers,max_convT_layers,max_skip_connections,input_shape):
		#Layers available for genetic alogorithm to make
		self.special_conv_layers = ['Conv2D','SeparableConv2D']
		self.special_convT_layers =['Conv2DTranspose','UpSampling2D']
		self.drop_layers = ['Dropout','SpatialDropout2D']#No 1
		self.merge_layers = ['Add']
		self.activations = ['"softmax"','"relu"','"elu"'] #ThresholdedReLU, LeakyReLU, PReLU should be added later
		self.pooling_layers = ['MaxPooling2D','AveragePooling2D']#No 2 ....we are not implementing these globalavg and globalmax due to dimension errors!
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
		self.padding_list = ['"same"'] #['same','valid']
		self.data_format_list = ['"channels_last"'] #['channels_last','channels_first']
		self.interpolation_list =['"bilinear"','"nearest"'] 
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
				self.a.append('BatchNormalization')
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
		self.genome.insert(0,[{'layer':'Input'}])
		self.genome.append([{'layer':'Dense',
							'units':self.pickone(self.units)}])

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
										
										'size':self.pickone(self.kernel_size_list),
										'data_format':self.pickone(self.data_format_list),
										'interpolation':self.pickone(self.interpolation_list)}
						self.genome[i][j] = self.gene
					
				if self.genome[i][j] == 'pooling_layers':
					if random.randint(0,1):
						self.gene = {'layer':'MaxPooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene
					
					else:
						self.gene = {'layer':'AveragePooling2D',
										'pool_size':self.pickone(self.pool_size_list),
										'strides':self.pickone(self.strides_list),
										'padding':self.pickone(self.padding_list),
										'data_format':self.pickone(self.data_format_list),
										}
						self.genome[i][j] = self.gene

				if self.genome[i][j] == 'BatchNormalization':
					self.gene = {'layer':'BatchNormalization',
								'momentum':self.pickone(self.momentum_list),
								'epsilon':self.pickone(self.epsilon_list)}
					self.genome[i][j] = self.gene


				if self.genome[i][j] == 'drop_layers':
					if random.randint(0,1):
						self.gene = {'layer':'Dropout',
										'rate':self.pickone(self.droputout_rate_list)}
						self.genome[i][j] = self.gene
					
					else:
						self.gene = {'layer':'SpatialDropout2D',
										'rate':self.pickone(self.droputout_rate_list),
										'data_format':self.pickone(self.data_format_list)}
						self.genome[i][j] = self.gene

		return self.genome 





	def decode_genome(self,genome,input_shape):
		#genome is a list of layers and their params
		self.commands = ['layer0 = Input(shape=np.array('+str(input_shape)+'))']
		self.layer_count = 1
		for i in range(1,len(self.genome),1):
			for j in self.genome[i]:
				self.cmd = 'layer'+str(self.layer_count)+'='
				self.params_lenght = (len(j)-1)
				self.params_coded = 0
				#print('\n')
				#print('\n')
				#print('\n')
				#print(j)#Debugging
				#print(self.params_lenght)#Debugging
				#_ = input()#Debugging
				for k in j:
					if k == 'layer':
						self.cmd = self.cmd + j[k] + '('
						self.layer_count = self.layer_count + 1
					else:
						if (self.params_coded == (self.params_lenght-1)):
							if (self.params_lenght == 1):
								self.cmd = self.cmd + k+ '=' + str(j[k]) + ')(layer' +str(self.layer_count-2)+')'
							else:
								self.cmd = self.cmd + ','+ k+ '=' + str(j[k]) + ')(layer' +str(self.layer_count-2)+')'
						else:
							if self.params_coded==0:
								self.cmd = self.cmd + k + '=' + str(j[k])
								self.params_coded = self.params_coded +1
							else:
								self.cmd = self.cmd + ',' + k + '=' + str(j[k])
								self.params_coded = self.params_coded +1
				self.commands.append(self.cmd)

		self.layer_count= self.layer_count-1 # name of the last genetic layer will be layer<self.layer_count>

		#Creation of actual keras model begins here!
		print('\n')
		print('Commands thats going to be executed now !')
		print('\n')
		print('\n')
		print('Commands thats going to be executed now !')
		for i in range(len(self.commands)):
			try:
				print(self.commands[i])
				exec(self.commands[i])
			except ValueError :
				print('Error at layer ',i)### This is done like this to make the error easy to see in the terminal :-p
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				print('Error at layer ',i)
				break
			self.layer_count = i
		exec('self.model = Model(input=layer0,output=layer'+str(self.layer_count)+')')
		return self.model


	def crossover(self,top_2_individuals):
		self.genome_1 = top_2_individuals[0]
		self.genome_2 = top_2_individuals[1]
		self.genome_1 = self.genome_1[1:(len(self.genome_1)-1)]
		self.genome_2 = self.genome_2[1:(len(self.genome_2)-1)]
		self.crossed_breeds = []
		for i in range(1,10,1):
			self.layer_cut_1 = math.floor((i/(10-i))*len(self.genome_1))
			self.layer_cut_2 = math.floor((i/(10-i))*len(self.genome_2))
			self.new_genome_1 = self.genome_2[:self.layer_cut_2]
			self.new_genome_2 = self.genome_1[:self.layer_cut_1]
			self.new_genome_1.extend(self.genome_1[(self.layer_cut_1):])
			self.new_genome_2.extend(self.genome_2[(self.layer_cut_2):])
			self.new_genome_1.insert(0,top_2_individuals[1][0])
			self.new_genome_2.insert(0,top_2_individuals[0][0])
			self.new_genome_1.append(top_2_individuals[0][-1])
			self.new_genome_2.append(top_2_individuals[1][-1])
			self.crossed_breeds.append(self.new_genome_1)
			self.crossed_breeds.append(self.new_genome_2)
		return self.crossed_breeds





	def mutation(self):
		pass






a = GA(20,15,0,(640,420,3))
genome_1 = a.init_genome()
genome_2 = a.init_genome()
crossed_breeds = a.crossover([genome_1,genome_2])
for i in crossed_breeds:
	for j in i:
		for k in j:
			print(k)

	print("New child")
	print('\n')
#model = a.decode_genome(a.init_genome(),a.input_shape)
#model.summary()