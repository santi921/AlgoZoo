import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import model_from_json


#custom metric implemented for tensorflow
def recall_m( y_true, y_pred):
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	        recall = true_positives / (possible_positives + K.epsilon())
	        return recall

#custom metric implemented for tensorflow
def precision_m( y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

#custom metric implemented for tensorflow
def f1_m( y_true, y_pred):
    precision = precision_m( y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))	

#custom cost function for keras
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

#custom cost function for keras
def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def stats(clf,image_vector, label_vector, X_train, X_test, y_train, y_test, train_time, tf_cond):
	
	#predict function for tf is different
	if (tf_cond == False): 
		#probs = clf.predict_proba(X_test)
		y_pred = clf.predict(X_test)
		train_pred = clf.predict(X_train)
	else: 
		y_pred = clf.predict_classes(X_test).astype('int').flatten()
		train_pred = clf.predict_classes(X_train).astype('int').flatten()
		
	#	preds = probs[:,1]
	parms = {}

	#accuracy as a measure of goodness
	#print(clf.predict(X_train))
	
	parms['train_acc'] = accuracy_score(train_pred.round(), y_train.round())
	parms['test_acc'] = accuracy_score(y_test.round(), y_pred.round())
	
	#confusion matrix for goodness with 0.5 accuracy
	conf = confusion_matrix(y_test.round(), y_pred.round())
	conf2 = confusion_matrix(clf.predict(X_train).round(), y_train.round())
	
	prec_test = conf[1][1]/ (conf[0][0] + conf[1][1])
	rec_test = conf[1][1]/ (conf[0][0] + conf[1][1])
	prec_train = conf2[1][1]/ (conf2[0][0] + conf2[1][1])
	rec_train = conf2[1][1]/ (conf2[0][0] + conf2[1][1])

	parms["f1m_test"] = 2 * (prec_test * rec_test)/(rec_test + prec_test)
	parms["f1m_train"] = 2 * (prec_train * rec_train)/(rec_train + prec_train)


	#to save on this computationally expensive step if the model is not accurate 
	cond = (parms['train_acc'] > 0.8 and tf_cond == False and train_time < 500)
	if(cond):	
		print("k-fold, slow performance")
		scores = cross_val_score(clf, image_vector, label_vector, cv=6)
		parms["kfolds"] = scores.mean()
	else: 
		#dummy val
		parms["kfolds"] = 0 

	return parms

def nn_generator(thickness, layers, flat = True):
	#some non augmented verions default to adding the flattening layer
	if (flat == True):
		model = keras.Sequential([keras.layers.Flatten(data_format="channels_last")])
	else: 
		model = keras.Sequential()
	
	#parameters for layer relations
	width = np.random.randint(low = 32, high = 256)
	scalar = np.random.randint(low = 1, high = 10)/6
	
	#for more compact network structures
	if(thickness != "wide"):
		width = int(width/2)


	parameter_list = [width, scalar]

	#iteratively add layers to the model
	for i in range(layers):
		#regularizer was useful
		model.add(keras.layers.Dense(int(width), kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Activation('relu'))
		#
		model.add(keras.layers.Dropout(0.3))
		width = int(width * scalar)

	model.add(keras.layers.Dense(1, kernel_initializer='normal'))
	model.add(keras.layers.Activation("sigmoid"))

	return model, parameter_list

def cnn_basic(size):
	model = keras.Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(keras.layers.Conv2D(16, (10, 10), activation='relu', input_shape=(size, size, 1)))
	model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Conv2D(16, (2, 2), activation='relu'))
	model.add(keras.layers.Conv2D(16, (2, 2), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd)

	#model.fit(x_train, y_train, batch_size=32, epochs=10)
	#score = model.evaluate(x_test, y_test, batch_size=32)

	return model


def lenet(size):
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation="tanh", input_shape=(size,size,1), padding="same"))
	model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))
	model.add(keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid"))
	model.add(keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
	model.add(keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation="tanh", padding="valid"))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(84, activation="tanh"))
	model.add(keras.layers.Dense(1, activation="softmax"))

	return model


d
