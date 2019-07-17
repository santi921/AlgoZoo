

import sys
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import svm, linear_model, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.naive_bayes import ComplementNB

from tensorflow import keras
from keras import regularizers
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator

#from io import StringIO
from ML_Helpers import stats, f1_loss, f1, f1_m, precision_m, recall_m
import Augmentor 

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
		self.dataset = results.dataset
		self.aug = results.aug
		self.imsize = results.imsize
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
			elif(self.type_train == "xgb"):
				self.xgboost_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "cnn"):
				self.cnn_method(image_vector, label_vector, self.parameters)
			elif(self.type_train == "rf"):
				self.random_forest_method(image_vector, label_vector, self.parameters)
			else:
				print("INVALID MODEL TYPE")
	####idea: create a global accuracy rating, only save models that pass a threshold or are
	####recordholding 

#######in the future send these to a help class and import
	def save_model_to_db(self, name, parms):
		print("save model statistics, history, etc, type")

	def parameter_generator(self):
		print("generates parameters for models")

	def time_string(self):
		now = datetime.datetime.now()
		time_nice = str(now.year) +"-"+ str(now.month) +"-"+ str(now.day) +"-"+ str(now.hour) +"-"+ str(now.minute) +"-" + str(now.second)
		return time_nice
		
	#use case
	#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])
	#loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)
	


	def random_forest_method(self, image_vector, label_vector, parameters):
		# n_estimators 			number of estimators, def 50
		# max_depth 			depth of decision trees, def 5
		# max_features 			number of features, def 1
		# verbose 				def 0
		

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			label_vector = np.rint(label_vector)
			print("percent positive: " + str(np.sum(label_vector)/len(label_vector)))

		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)

		clf = RandomForestClassifier(n_estimators = 50, n_jobs=10)
		clf.fit(X_train,y_train)

		params = self.stats(clf, image_vector, label_vector, X_train, X_test, y_train, y_test)
		
		name = "AlgorithmDB/rf_" + self.time_string() + ".pkl"
		#joblib.dump(clf, name)
		#params["type"] = "sklearn"
		#self.save_model_to_db(name, parms)


	def nn_method(self, image_vector, label_vector, parameters):
		# l2
		# loss
		# dropout 
		# opt 
		# batch_size
		# epochs 
		# activations 
		
		batch_size = 1
		
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

		model = keras.Sequential([
	    keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
   	    keras.layers.Dropout(0.3),
	    keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
	   	keras.layers.Dropout(0.4),
	    keras.layers.Dense(16, activation="relu"),
	    keras.layers.Dropout(0.3),
	    keras.layers.Dense(1, activation="sigmoid")
		])

		opt = "adam"	
		loss_func = f1_loss
		loss_func = 'binary_crossentropy'
		model.compile(optimizer=opt, loss= loss_func,  metrics=['acc',f1_m, precision_m, recall_m])
		#model.compile(optimizer=opt, loss= loss_func,  metrics=['accuracy'])
	


		label_vector = label_vector.astype(int)
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)
		if(self.aug == True):


			print("image shape: " + str(X_train[np.newaxis].shape))
			print("label shape: " + str(y_train.shape))
			
			datagen = ImageDataGenerator(rotation_range=360, zoom_range=0.1, data_format = "channels_first")
			#datagen.fit(X_train)
			it = datagen.flow(X_train[np.newaxis].reshape(len(y_train), self.imsize,self.imsize,1), y_train, batch_size = batch_size)
			history = model.fit_generator(it, steps_per_epoch = len(y_train) / batch_size, verbose = self.verbose, validation_data=(X_test, y_test))

		else: 
			print("aug false")

			#y_train = Augmentor.Pipeline.categorical_labels(y_train.astype(int))
			#y_test = Augmentor.Pipeline.categorical_labels(y_test.astype(int))
			history = model.fit(X_train,y_train , batch_size=batch_size, epochs=200, verbose = self.verbose)



		name = "AlgorithmDB/nn_" + self.time_string() + ".pkl"


		###figure out tensofrlow statistics
		#self.save_model_to_db(name, parms)

		#params["type"] = "tensorflow"

		#plt.plot(history.history["accuracy"])
		#plt.xlabel("epochs")
		#plt.ylabel("accuracy")
		#plt.title("Training Curve")
		#plt.show()
		#score = model.evaluate(X_test, y_test) 
		#print(score)

	def sgd_method(self, image_vector, label_vector, parameters):
		
		# loss 			- "hinge", "log", "square_hinge", "preceptron", "squared loss"
		# max_iter 		- epochs
		# alpha 		- 0.0001
		# verbose 		- 0
		# tol 			- 1e-3, tolerances before premature stop




		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			#0.2 for eh results on dataset 2/3
			label_vector = np.rint(label_vector + 0.2)
			
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)


		clf = linear_model.SGDClassifier(loss = 'log',alpha = 0.000000001, max_iter=100000, tol=1e-8, early_stopping= False)	
		clf.fit(X_train, y_train)
		
		params = stats(clf, image_vector, label_vector, X_train, X_test, y_train, y_test)
		
		#name = "AlgorithmDB/sgd_" + self.time_string() + ".pkl"
		#joblib.dump(clf, name)
		#self.save_model_to_db(name, parms)
		#params["type"] = "sklearn"





	

	def svm_method(self, image_vector, label_vector, parameters):
		###parameters####
		# C 				- penalty def 1.0
		# kernel function 	- linear, rbf, poly, sigmoid, precomputed
		# degree			- of poly def = 3
		# gamma 			- float, def is 1/features for rbf, ,poly, sigmoid 
		# coef0				- def 0.0 for poly, sigmoid
		#tol 				- early stopping, def = 1e-3
		#class_weight		- use to adjust class weight, USE {dict, 'balanced'}
		#verbose			- def 0 

		

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			label_vector = np.rint(label_vector)
			print("percent positive: " + str(np.sum(label_vector)/len(label_vector)))

		clf = svm.SVC(verbose = 1, kernel = "linear", probability = True)
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)	
		clf.fit(X_train, y_train)
		params = stats(clf, image_vector, label_vector, X_train, X_test, y_train, y_test)

		#params["type"] = "sklearn"


	def xgboost_method(self):
		print("boost")



		#params["type"] = "sklearn"


	


	def cnn_method(self, image_vector, label_vector, parameters):
		print("CNN")
		print("add options for transfer learning")



		#params["type"] = "tensorflow"
