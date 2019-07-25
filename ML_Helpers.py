import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras import regularizers

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

def stats(clf,image_vector, label_vector, X_train, X_test, y_train, y_test, train_time):

	y_pred = clf.predict(X_test)
	probs = clf.predict_proba(X_test)
	preds = probs[:,1]
	#rf parameter definition 
	parms = {}

	#accuracy as a measure of goodness
	
	parms['train_acc'] = accuracy_score(clf.predict(X_train), y_train)
	parms['test_acc'] = accuracy_score(y_test, y_pred)
	
	#confusion matrix for goodness with 0.5 accuracy
	conf = confusion_matrix(y_test, y_pred)
	conf2 = confusion_matrix(clf.predict(X_train), y_train)
	#print("test confusion: "+ str(conf))
	#print("train confusion: " + str(conf2))
	#parms["false_poss_train_50"] =conf2[0][1]
	#parms["false_neg_train_50"] = conf2[1][0]
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	prec_test = conf[1][1]/ (conf[0][0] + conf[1][1])
	rec_test = conf[1][1]/ (conf[0][0] + conf[1][1])
	prec_train = conf2[1][1]/ (conf2[0][0] + conf2[1][1])
	rec_train = conf2[1][1]/ (conf2[0][0] + conf2[1][1])

	parms["f1m_test"] = 2 * (prec_test * rec_test)/(rec_test + prec_test)
	parms["f1m_train"] = 2 * (prec_train * rec_train)/(rec_train + prec_train)


	#to save on this computationally expensive step if the model is not accurate 
	cond = (parms['train_acc'] > 0.6 and parms['test_acc'] > 0.6 and train_time < 500)
	if(cond):	
		print("k-fold, slow performance")
		scores = cross_val_score(clf, image_vector, label_vector, cv=6)
		parms["kfolds"] = scores.mean()
	else: 
		#dummy val
		parms["kfolds"] = 0 

	return parms

def nn_generator(thickness, layers, flat = True):
	
	if (flat == True):
		model = keras.Sequential([keras.layers.Flatten(data_format="channels_last")])
	else: 
		model = keras.Sequential()
	
	width = np.random.randint(low = 64, high = 256)
	scalar = np.random.randint(low = 1, high = 10)/5

	#for testing
	width = 10
	scalar = 1

	if(thickness == "wide"):
		width *= 4
	parameter_list = [width, scalar]


	for i in range(layers):
		model.add(keras.layers.Dense(width,activation = keras.layers.Activation('relu'), kernel_regularizer=regularizers.l2(0.001)))
		model.add(keras.layers.Dropout(0.3))
		width = int(width * scalar)

	model.add(keras.layers.Dense(1))
	model.add(keras.layers.Activation('relu'))

	return model, parameter_list

def cnn_basic(size):
	model = keras.Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.
	model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(size, size, 1)))
	model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
	model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
	model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(1, activation='softmax'))

	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	#model.compile(loss='categorical_crossentropy', optimizer=sgd)

	#model.fit(x_train, y_train, batch_size=32, epochs=10)
	#score = model.evaluate(x_test, y_test, batch_size=32)

	return model