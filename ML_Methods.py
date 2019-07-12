import numpy as np
import tensorflow as tf
import sys
import datetime
import matplotlib.pyplot as plt

from sklearn import svm 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc


from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras
from keras.utils import to_categorical
from keras import regularizers


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
		self.dataset = results.dataset
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



	def recall_m(self, y_true, y_pred):
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	        recall = true_positives / (possible_positives + K.epsilon())
	        return recall

	def precision_m(self, y_true, y_pred):
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	        precision = true_positives / (predicted_positives + K.epsilon())
	        return precision

	def f1_m(self, y_true, y_pred):
	    precision = self.precision_m( y_true, y_pred)
	    recall = self.recall_m(y_true, y_pred)
	    return 2*((precision*recall)/(precision+recall+K.epsilon()))	

	def stats(self, clf,image_vector, label_vector, X_train, X_test, y_train, y_test):

		y_pred = clf.predict(X_test)
		probs = clf.predict_proba(X_test)
		preds = probs[:,1]
		print(preds[0:50])
		#rf parameter definition 
		parms = {}

		#accuracy as a measure of goodness
		parms['accuracy_train'] = metrics.accuracy_score(clf.predict(X_train), y_train)
		parms['accuracy_trial'] = metrics.accuracy_score(y_test, y_pred)
		
		#confusion matrix for goodness with 0.5 accuracy
		conf = metrics.confusion_matrix(y_test, y_pred)
		conf2 = metrics.confusion_matrix(clf.predict(X_train), y_train)
		print("test confusion: "+ str(conf))
		print("train confusion: " + str(conf2))

		parms["false_poss_train_50"] =conf2[0][1]
		parms["false_neg_train_50"] = conf2[1][0]

		scores = cross_val_score(clf, image_vector, label_vector, cv=8)
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		parms["cross_score_mean"] = scores.mean()
		parms["cross_score_std"] = scores.std()
		
		fpr, tpr, threshold = roc_curve(y_test, preds)
		roc_auc = auc(fpr, tpr)

		print("AUC ROC: "+ str(roc_auc_score(y_test, preds)))
		# method I: plt
		
		plt.title('Receiver Operating Characteristic')
		plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.show()
		
		return parms


		





	def random_forest_method(self, image_vector, label_vector, parameters):
		# n_estimators = 50
		# max_depth = 5
		# max_features = 1
		# verbose = 0
		

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			label_vector = np.rint(label_vector)
			print("percent positive: " + str(np.sum(label_vector)/len(label_vector)))

		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)
		clf = RandomForestClassifier(n_estimators = 50, n_jobs=10)
		clf.fit(X_train,y_train)

		params = self.stats(clf, image_vector, label_vector, X_train, X_test, y_train, y_test)
		
		name = "AlgorithmDB/rf_" + self.time_string() + ".pkl"
		#joblib.dump(clf, name)
		#self.save_model_to_db(name, parms)

	def cnn_method(self, image_vector, label_vector, parameters):
		print("CNN")
		print("add options for transfer learning")

	def nn_method(self, image_vector, label_vector, parameters):
		
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)

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
		model.compile(optimizer=opt, loss='binary_crossentropy',  metrics=['acc',self.f1_m, self.precision_m, self.recall_m])

		history = model.fit(X_train,y_train , batch_size=50, epochs=20, verbose = self.verbose)

		name = "AlgorithmDB/nn_" + self.time_string() + ".pkl"


		###figure out tensofrlow statistics
		#self.save_model_to_db(name, parms)
		plt.plot(history.history["acc"])
		plt.xlabel("epochs")
		plt.ylabel("accuracy")
		plt.title("Training Curve")
		plt.show()
		score = model.evaluate(X_test, y_test) 
		print(score)

	def sgd_method(self, image_vector, label_vector, parameters):
		
		# loss = "hinge", "log", "square_hinge", "preceptron", "squared loss"
		# max_iter = epochs
		# alpha = 0.0001
		# verbose - 0
		# tol = 1e-3

		if (int(self.dataset) == 2 or int(self.dataset) == 3):
			#0.2 for eh results on dataset 2/3
			label_vector = np.rint(label_vector + 0.2)
			
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)


		clf = linear_model.SGDClassifier(loss = 'log',alpha = 0.000000001, max_iter=100000, tol=1e-8, early_stopping= False)	
		clf.fit(X_train, y_train)
		
		params = self.stats(clf, image_vector, label_vector, X_train, X_test, y_train, y_test)
		
		#name = "AlgorithmDB/sgd_" + self.time_string() + ".pkl"
		#joblib.dump(clf, name)
		#self.save_model_to_db(name, parms)






	def bayes_method(self):
		print("bayes")
	
	def xgboost_method(self):
		print("boost")

	def svm_method(self, image_vector, label_vector, parameters):
		###parameters####
		#C 					- penalty def 1.0
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

		clf = svm.SVC(verbose = 1, kernel = "linear")
		print("classifier created")
		X_train, X_test, y_train, y_test = train_test_split(image_vector, label_vector, test_size=0.2)
		print("data split")
		clf.fit(X_train, y_train)
		print("model trained")
		print("Accuracy on test metric: " + str(metrics.accuracy_score(clf.predict(X_test), y_test)))
		scores = cross_val_score(clf, image_vector, label_vector, cv=8)
		print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def robust_pca_method(self, image_vector, label_vector, parameters):
		print("PCA")