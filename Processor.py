import argparse
import ML_Methods

def imaging_processor():

	print("something")
	#separate images, collect 
	#gather db
	#export image vectors to 

if  __name__=="__main__":
	#arguement parsing for options 
	parser = argparse.ArgumentParser(description='Model Selector and Parameters')

	parser.add_argument("-v","--verbose", action = 'store_true' , dest= 'verbose', default = False,  help = "increase output printing")
	parser.add_argument("--svc", action = 'store_true' , dest= 'svc', default = False,  help = "increase output printing")
	parser.add_argument("--svm", action = 'store_true' , dest= 'svm', default = False,  help = "increase output printing")
	parser.add_argument("--nn", action = 'store_true' , dest= 'nn', default = False,  help = "increase output printing")
	parser.add_argument("--cnn", action = 'store_true' , dest= 'cnn', default = False,  help = "increase output printing")
	parser.add_argument("--robust",action = 'store_true' , dest= 'robust', default = False,  help = "increase output printing")
	parser.add_argument("--gans",action = 'store_true' , dest= 'gans', default = False,  help = "increase output printing")
	parser.add_argument("--custom",action = 'store_true' , dest= 'custom', default = False,  help = "increase output printing")

	parser.add_argument("-iter", action = "store", dest = "iterations",default = 1,  help= "number of models")

	results = parser.parse_args()
	ml = ML_Methods.Methods(results.iterations , results.custom)
	
	
	print("______________Options Selected_____________")
	print("verbose: \t\t"+str(results.verbose))
	print("# of models trained:\t"+ str(results.iterations))
	print("custom parameters? \t" + str(results.custom))
	print("verbose?\t\t" + str(results.verbose))
	print("__________________________________________")
	if(results.verbose):
		if(results.svc):
			print("SVC model selected..")
			ml.svc_method()


		elif(results.svm):
			print("SVC model selected...")
			ml.svm_method()

		elif(results.nn):
			print("NN model selected...")
			ml.nn_method()

		elif(results.cnn):
			print("CNN model selected...")
			ml.cnn_method()

		elif(results.robust):
			print("RobustPCA model selected...")
			ml.robust_pca_method()

		elif(results.gans):
			print("Loading Generative Network...")
			ml.gans_method()

		else:
			print("Defaulting to CNNs...")
			ml.cnn_method()
	










