import numpy as np
import tensorflow as tf
import sys

from sklearn import svm 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from tensorflow import keras
from keras.utils import to_categorical
from keras import regularizers
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

#from io import StringIO

class Methods: 

	def __init__(self, results):
		self.results = results
		self.iterations = results.iterations
		self.type_train = results.type_train
		self.parameters = results.parameters
		self.epochs = results.epochs
		self.verbose = results.verbose
		self.dataset = results.dataset
		self.transfer = results.transfer
		#opens global database of models that have worked 

	

	def iterator(self, image_vector, label_vector):
		#add reasonable parameter ranges for all of these 
		for i in range(self.iterations):
			
			if(self.type_train == "nn"):
				self.nn_method(image_vector, label_vector, self.parameters )
			elif(self.type_train == "sgd"):
				self.sgd_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "svm"):
				self.svm_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "robust_pca"):
				self.robust_pca_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "cnn"):
				self.cnn_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "gans"):
				self.gans_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "bayes"):
				self.bayes_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "rf"):
				self.random_forest_method(image_vector, label_vector, self.parameters)
			else:
				print("INVALID MODEL TYPE")
	####idea: create a global accuracy rating, only save models that pass a threshold or are
	####recordholding 


	def save_model():
		print("save model statistics, history, etc, type")

	def parameter_generator():
		print("generates parameters for models")
		
		
	def robust_pca_method(self, image_vector, label_vector, parameters):
		print("PCA")
	#def gans_method(self, image_vector, label_vector, parameters):
	#	print("GANS")	
	def cnn_method(self, image_vector, label_vector, parameters):
		print("CNN")
		print("add options for transfer learning")
	def bayes_method(self):
		print("bayes")
	def random_forest_method(self, image_vector, label_vector, parameters):

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			label_vector = np.rint(label_vector)

		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)
		clf = RandomForestClassifier(n_jobs = 2, random_state=0)
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

	def svm_method(self, image_vector, label_vector, parameters):
		print("SVM")
	
	
	def nn_method(self, image_vector, label_vector, parameters):
		
		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth=True
		#config.gpu_options.per_process_gpu_memory_fraction = 0.4

		#sess = tf.Session(config=config)


		# if (parameters.aug == True): path for 
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)

		model = keras.Sequential([
    	#keras.layers.Dense(512, kernel_initializer="normal",kernel_regularizer=regularizers.l2(0.0001),  activation="sigmoid", input_dim = X_train.shape[1]),
	    #keras.layers.Dense(, activation=tf.nn.relu),
	    #keras.layers.Dense(256,kernel_regularizer=regularizers.l2(0.001), activation="relu"),
	    #keras.layers.Dropout(0.2),
	    #keras.layers.Dense(, activation="relu"),
	    keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
   	    keras.layers.Dropout(0.3),
	    keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
	   	keras.layers.Dropout(0.4),
	    keras.layers.Dense(16, activation="relu"),
	    keras.layers.Dropout(0.3),
	    keras.layers.Dense(1, activation="sigmoid")
		])

		opt = "adam"
		model.compile(optimizer=opt, loss='binary_crossentropy',  metrics=['accuracy'])

		history = model.fit(X_train,y_train , batch_size=10, epochs=200, verbose = self.verbose)
		plt.plot(history.history["acc"])
		plt.xlabel("epochs")
		plt.ylabel("accuracy")
		plt.title("Training Curve")
		plt.show()
		score = model.evaluate(X_test, y_test) #89% thus far

	def sgd_method(self, image_vector, label_vector, parameters):
		# loss = "hinge", "log", "square_hinge", "preceptron", "squared loss"
		# max_iter = epochs
		# alpha = 0.0001
		# verbose - 0
		# tol = 1e-3

		#sdg hates non-integer labels
		#think about this more

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			label_vector = np.rint(label_vector + 0.2)
			print("rounds here")

		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)

		

		iterations = []
		accuracy_test = []
		
		
		for n in range(10):
			clf = linear_model.SGDClassifier(loss = 'squared_loss',alpha = 0.000000001, max_iter=10**(n/2), tol=1e-8, early_stopping= False)	
			clf.fit(X_train, y_train)
			accuracy_test.append(clf.score(X_test, y_test))
			iterations.append(10**(n/2))
		plt.figure()
		plt.plot(iterations, accuracy_test)
		plt.xlabel("training iterations")
		plt.ylabel("Test Accuracy")
		plt.xscale('log')
		plt.title("Training accuracy over iterations")
		plt.show()
