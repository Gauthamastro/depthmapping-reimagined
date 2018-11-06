import random,math
from random import shuffle
from keras.layers import Conv2D,SeparableConv2D,Conv2DTranspose,UpSampling2D,Dense,Dropout,SpatialDropout2D,Concatenate,Add
from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization,AveragePooling2D,GlobalMaxPooling2D,MaxPooling2D,GlobalAveragePooling2D
import numpy as np 
from collections import Counter

seed =34
random.seed(34)

#We give optimal parameters in the init fn of these classes !
#these will be accessed on execution for mutation and crossover!

class GA:
	def __init__(self,max_conv_layers,max_convT_layers,max_individuals,input_shape,x_train,y_train,x_test,y_test,epochs):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test

		self.epochs = epochs

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
		self.max_individuals=max_individuals
		self.input_shape = input_shape	

		#Params that genetic algorithm will use to search!
		self.output_filter_depth_list = [2**i for i in range(1,11,1)] # assumeing channels last
		self.kernel_size_list = [1,3,5,7,9,11]
		self.strides_list = [1,2,3,4,5]
		self.padding_list = ['"same"','"valid"'] #['same','valid']
		self.data_format_list = ['"channels_last"'] #['channels_last','channels_first']
		self.interpolation_list =['"bilinear"','"nearest"'] 
		self.momentum_list = [0.9,0.99,0.999]
		self.epsilon_list = [0.001,0.01,0.1]
		self.pool_size_list = [1,2,3]
		self.units = [2**i for i in range(1,12,1)]
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
		for i in range(1,len(genome),1):
			for j in genome[i]:
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
		#exec('self.model = Model(input=layer0,output=layer'+str(self.layer_count)+')')
		return self.layer_count


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

	def  max_count(self,l):
		self.data = Counter(l)
		#print(type(self.data.most_common(1)[0][0]))
		return self.data.most_common(1)[0][0]

	def best_params(self,best_5):
		self.Conv2D_params_list = []
		self.SeparableConv2D_params_list = []
		self.Conv2DTranspose_params_list= []
		self.UpSampling2D_params_list = []
		self.BatchNormalization_params_list = []
		self.Dropout_params_list = []
		self.SpatialDropout2D_params_list = []
		self.MaxPooling2D_params_list = []
		self.AveragePooling2D_params_list = []
		for model in best_5:
			for module in model:
				for genome in module:
					if genome['layer']== 'Conv2D':
						self.Conv2D_params_list.append([genome['filters'],genome['kernel_size'],genome['strides'],genome['padding'],genome['data_format'],genome['activation']])
					if genome['layer']== 'SeparableConv2D':
						self.SeparableConv2D_params_list.append([genome['filters'],genome['kernel_size'],genome['strides'],genome['padding'],genome['data_format'],genome['activation']])
					if genome['layer']== 'Conv2DTranspose':
						self.Conv2DTranspose_params_list.append([genome['filters'],genome['kernel_size'],genome['strides'],genome['padding'],genome['data_format'],genome['activation']])
					if genome['layer']== 'UpSampling2D':
						self.UpSampling2D_params_list.append([genome['size'],genome['data_format'],genome['interpolation']])
					if genome['layer']== 'BatchNormalization':
						self.BatchNormalization_params_list.append([genome['momentum'],genome['epsilon']])
					if genome['layer']== 'Dropout':
						self.Dropout_params_list.append([genome['rate']])
					if genome['layer']== 'SpatialDropout2D':
						self.SpatialDropout2D_params_list.append([genome['rate'],genome['data_format']])
					if genome['layer']== 'MaxPooling2D':
						self.MaxPooling2D_params_list.append([genome['pool_size'],genome['strides'],genome['padding'],genome['data_format']])
					if genome['layer']== 'AveragePooling2D':
						self.AveragePooling2D_params_list.append([genome['pool_size'],genome['strides'],genome['padding'],genome['data_format']])
		
		self.Conv2D_params_list = np.array(self.Conv2D_params_list)
		self.SeparableConv2D_params_list = np.array(self.SeparableConv2D_params_list)
		self.Conv2DTranspose_params_list= np.array(self.Conv2DTranspose_params_list)
		self.UpSampling2D_params_list = np.array(self.UpSampling2D_params_list)
		self.BatchNormalization_params_list = np.array(self.BatchNormalization_params_list)
		self.Dropout_params_list = np.array(self.Dropout_params_list)
		self.SpatialDropout2D_params_list = np.array(self.SpatialDropout2D_params_list)
		self.MaxPooling2D_params_list = np.array(self.MaxPooling2D_params_list)
		self.AveragePooling2D_params_list = np.array(self.AveragePooling2D_params_list)

		self.Conv2D_params_list = [self.max_count(self.Conv2D_params_list[:,0]),self.max_count(self.Conv2D_params_list[:,1]),self.max_count(self.Conv2D_params_list[:,2]),self.max_count(self.Conv2D_params_list[:,3]),self.max_count(self.Conv2D_params_list[:,4]),self.max_count(self.Conv2D_params_list[:,5])]
		self.SeparableConv2D_params_list = [self.max_count(self.SeparableConv2D_params_list[:,0]),self.max_count(self.SeparableConv2D_params_list[:,1]),self.max_count(self.SeparableConv2D_params_list[:,2]),self.max_count(self.SeparableConv2D_params_list[:,3]),self.max_count(self.SeparableConv2D_params_list[:,4]),self.max_count(self.SeparableConv2D_params_list[:,5])]
		self.Conv2DTranspose_params_list = [self.max_count(self.Conv2DTranspose_params_list[:,0]),self.max_count(self.Conv2DTranspose_params_list[:,1]),self.max_count(self.Conv2DTranspose_params_list[:,2]),self.max_count(self.Conv2DTranspose_params_list[:,3]),self.max_count(self.Conv2DTranspose_params_list[:,4]),self.max_count(self.Conv2DTranspose_params_list[:,5])]
		self.UpSampling2D_params_list =[self.max_count(self.UpSampling2D_params_list[:,0]),self.max_count(self.UpSampling2D_params_list[:,1]),self.max_count(self.UpSampling2D_params_list[:,2])]
		self.BatchNormalization_params_list = [self.max_count(self.BatchNormalization_params_list[:,0]),self.max_count(self.BatchNormalization_params_list[:,1])]
		self.Dropout_params_list = [self.max_count(self.Dropout_params_list[:,0])]
		self.SpatialDropout2D_params_list = [self.max_count(self.SpatialDropout2D_params_list[:,0]),self.max_count(self.SpatialDropout2D_params_list[:,1])]
		self.MaxPooling2D_params_list = [self.max_count(self.MaxPooling2D_params_list[:,0]),self.max_count(self.MaxPooling2D_params_list[:,1]),self.max_count(self.MaxPooling2D_params_list[:,2]),self.max_count(self.MaxPooling2D_params_list[:,3])]
		self.AveragePooling2D_params_list = [self.max_count(self.AveragePooling2D_params_list[:,0]),self.max_count(self.AveragePooling2D_params_list[:,1]),self.max_count(self.AveragePooling2D_params_list[:,2]),self.max_count(self.AveragePooling2D_params_list[:,3])]
		#print(self.Conv2D_params_list)

		self.best_params = [{'layer':'Conv2D',
										'filters':int(self.Conv2D_params_list[0]),
										'kernel_size':int(self.Conv2D_params_list[1]),
										'strides':int(self.Conv2D_params_list[2]),
										'padding':self.Conv2D_params_list[3],
										'data_format':self.Conv2D_params_list[4],
										'activation':self.Conv2D_params_list[5]
										},
										 {'layer':'SeparableConv2D',
										'filters':int(self.Conv2D_params_list[0]),
										'kernel_size':int(self.Conv2D_params_list[1]),
										'strides':int(self.Conv2D_params_list[2]),
										'padding':self.Conv2D_params_list[3],
										'data_format':self.Conv2D_params_list[4],
										'activation':self.Conv2D_params_list[5]
										},
										 {'layer':'Conv2DTranspose',
										'filters':int(self.Conv2D_params_list[0]),
										'kernel_size':int(self.Conv2D_params_list[1]),
										'strides':int(self.Conv2D_params_list[2]),
										'padding':self.Conv2D_params_list[3],
										'data_format':self.Conv2D_params_list[4],
										'activation':self.Conv2D_params_list[5]
										},
										{'layer':'UpSampling2D',
										'size':int(self.UpSampling2D_params_list[0]),
										'data_format':self.UpSampling2D_params_list[1],
										'interpolation':self.UpSampling2D_params_list[2]
										},
										{'layer':'MaxPooling2D',
										'pool_size':int(self.MaxPooling2D_params_list[0]),
										'strides':int(self.MaxPooling2D_params_list[1]),
										'padding':self.MaxPooling2D_params_list[2],
										'data_format':self.MaxPooling2D_params_list[3],
										},
										{'layer':'AveragePooling2D',
										'pool_size':int(self.AveragePooling2D_params_list[0]),
										'strides':int(self.AveragePooling2D_params_list[1]),
										'padding':self.AveragePooling2D_params_list[2],
										'data_format':self.AveragePooling2D_params_list[3],
										},
										{'layer':'BatchNormalization',
										'momentum':self.BatchNormalization_params_list[0],
										'epsilon':self.BatchNormalization_params_list[1]
										},
										{'layer':'Dropout',
										'rate':self.Dropout_params_list[0]
										},
										{'layer':'SpatialDropout2D',
										'rate':float(self.SpatialDropout2D_params_list[0]),
										'data_format':self.SpatialDropout2D_params_list[1]
										}]
		return self.best_params

	def mutation(self,next_best_5):
		#here the best 5 means next 5 after top 2 
		# ie from 3rd to 7th best model
		shuffle(next_best_5)
		self.genome_1 = next_best_5[0]
		self.genome_2 = next_best_5[1]
		self.best_params_list = self.best_params(next_best_5)
		#Code for the 19th person
		for module in self.genome_1:
			for i in range(len(module)):
				self.gene = module[i]
				if self.gene['layer'] == 'Conv2D':
					module[i] = self.best_params_list[0]
				if self.gene['layer'] == 'SeparableConv2D':
					module[i] = self.best_params_list[1]
				if self.gene['layer'] == 'Conv2DTranspose':
					module[i] = self.best_params_list[2]
				if self.gene['layer'] == 'UpSampling2D':
					module[i] = self.best_params_list[3]
				if self.gene['layer'] == 'MaxPooling2D':
					module[i] = self.best_params_list[4]
				if self.gene['layer'] == 'AveragePooling2D':
					module[i] = self.best_params_list[5]
				if self.gene['layer'] == 'BatchNormalization':
					module[i] = self.best_params_list[6]
				if self.gene['layer'] == 'Dropout':
					module[i] = self.best_params_list[7]
				if self.gene['layer'] == 'SpatialDropout2D':
					module[i] = self.best_params_list[8]

		#Code for the 20th person!
		self.mutated_layer_num = random.randint(0,len(self.genome_2))
		self.mutated_layer = self.genome_2[self.mutated_layer_num]
		#print(self.mutated_layer)##For debugging
		for i in self.mutated_layer:
			for j in i:
				if j == 'rate':
					while True :
						self.a = self.pickone(self.droputout_rate_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.droputout_rate_list)
				if j == 'momentum':
					while True :
						self.a = self.pickone(self.momentum_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.momentum_list)
				if j == 'epsilon':
					while True :
						self.a = self.pickone(self.epsilon_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.epsilon_list)
				if j == 'strides':
					while True :
						self.a = self.pickone(self.strides_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.strides_list)
				if j == 'activation':
					while True :
						self.a = self.pickone(self.activations)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.activations)
				if j == 'kernel_size':
					while True :
						self.a = self.pickone(self.kernel_size_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.kernel_size_list)
				if j == 'filters':
					while True :
						self.a = self.pickone(self.output_filter_depth_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.output_filter_depth_list)
				if j == 'interpolation':
					while True :
						self.a = self.pickone(self.interpolation_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.interpolation_list)
				if j == 'size':
					while True :
						self.a = self.pickone(self.kernel_size_list)
						if self.a != i[j] :
							i[j] = self.a
							break
						else:
							self.a = self.pickone(self.kernel_size_list)
		#print(self.mutated_layer)#For debugging
		self.genome_2[self.mutated_layer_num] = self.mutated_layer

		#self.genome_2 is ready!
		#genome_1 is  ready
		return self.genome_1,self.genome_2

	def _handle_broken_model(self, model, error):
        del model
        gc.collect()
        if K.backend() == 'tensorflow':
            K.clear_session()
            tf.reset_default_graph()

        print('An error occurred and the model could not train:')
        print('\n')
        print(error)
        print('\n')
        print(('Please ensure that your model'
               'constraints live within your computational resources.'))

	def evaluate_genome(self,model, epochs,Generation,individual):
        loss, accuracy = None, None
        try:
            model.fit(self.x_train, self.y_train,
                      validation_data=(self.x_test, self.y_test),
                      epochs=epochs,
                      verbose=1,
                      callbacks=[
                          EarlyStopping(monitor='val_loss',
                                        patience=1,
                                        verbose=1)
                      ])
            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        except Exception as e:
            loss, accuracy = self._handle_broken_model(model, e)
        self.name = 'generation-'+str(Generation)+'-individual-'+str(individual)
        return [self.name,model, loss, accuracy]

	def init_population(self):
		self.pop = []
		for i in range(self.max_individuals):
			self.pop.append(self.init_genome())
		return self.pop

	def maintain_pop(self,pop):
		self.Generation = 1
		self.pop = pop
		while self.Generation <= self.max_generations:
			print('Generation Count = ',i)
			self.scores = []
			for i in range(len(self.pop)):
				self.layer_count = self.decode_genome(i)
				'''
				include the data specific final layers using exec commands also take care of the layer number

				'''
				exec('self.model = Model(input=layer0,output=layer'+str(self.layer_count)+')')
				if self.layer_count>20:
					self.adm = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
					self.model.compile(loss='binary_crossentropy',optimizer=self.adm,metrics=['accuracy']) #loss is not defined
				else:
					self.sgd =keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=True)
					self.model.compile(loss='binary_crossentropy',optimizer=self.sgd,metrics=['accuracy'])#loss is not defined!
				self.scores.append(evaluate_genome(self.model,self.epochs,self.Generation,i))
			self.best_7 = []
			for i in self.scores:
				pass #code the rest from here!
			self.Generation = self.Generation + 1


			
a = GA(20,15,20,(640,420,3))#Data is not given yet!

members = a.init_population()
a.maintain_pop(members)

#model = a.decode_genome(a.init_genome(),a.input_shape)
#model.summary()