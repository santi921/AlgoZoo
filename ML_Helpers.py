import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def recall_m( y_true, y_pred):
	        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	        recall = true_positives / (possible_positives + K.epsilon())
	        return recall

def precision_m( y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m( y_true, y_pred):
    precision = precision_m( y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))	




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

def stats(clf,image_vector, label_vector, X_train, X_test, y_train, y_test):

	y_pred = clf.predict(X_test)
	probs = clf.predict_proba(X_test)
	preds = probs[:,1]
	print(preds[0:50])
	#rf parameter definition 
	parms = {}

	#accuracy as a measure of goodness
	parms['accuracy_train'] = accuracy_score(clf.predict(X_train), y_train)
	parms['accuracy_trial'] = accuracy_score(y_test, y_pred)
	
	#confusion matrix for goodness with 0.5 accuracy
	conf = confusion_matrix(y_test, y_pred)
	conf2 = confusion_matrix(clf.predict(X_train), y_train)
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
