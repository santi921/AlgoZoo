import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score


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
