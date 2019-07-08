import Augmentor 
import numpy as np 
import matplotlib.pyplot as plt
import os
from PIL import Image 
import pandas as pd

 
def aug_A():
	stats = pd.read_pickle("./StandardizedImages/stats.pkl")
	dir_A = "./StandardizedImages/WholeTissue/"


	list_label_images_raw = os.listdir(dir)
	list_label_images= [] 
	list_images = []

	for index, row in stats.iterrows():

		scale_temp = row["scale_val"]
		temp_image = np.array(Image.open(dir + row["id"] + "standard.png")).repeat(scale_temp,axis=0).repeat(scale_temp,axis=1)
		for i in range(int(float(temp_image.shape[0])/float(size))):
				for j in range(int(float(temp_image.shape[1])/float(size))):
					print("temp")
	return list_images, list_label_images

def aug_B():
	dir_B = "./images/OnlineDatabases/DatasetB/labels_two/"
	list_label_images_raw = os.listdir(dir)
	list_label_images= [] 

	for i, j in enumerate(list_label_images_raw):

		if (j[0:3] =="two" and  j[-22:-19] == "Red"):
			file = str(j[8:-1] + j[-1])
			temp = np.array(Image.open(str(dir + j)))
			temp2 = np.array(Image.open(str(dir + file)))


def load_A(size = 128):
	stats = pd.read_pickle("./StandardizedImages/stats.pkl")
	dir = "./StandardizedImages/WholeTissue/"

	list_label_images_raw = os.listdir(dir)
	list_label_images= [] 
	list_images = []

	for index, row in stats.iterrows():

		scale_temp = row["scale_val"]
		temp_image = np.array(Image.open(dir + row["id"] + "standard.png")).repeat(scale_temp,axis=0).repeat(scale_temp,axis=1)
		for i in range(int(float(temp_image.shape[0])/float(size))):
				for j in range(int(float(temp_image.shape[1])/float(size))):
					
					list_label_images.append(row["labels"])
					list_images.append(temp_image[i:i+size,j:j+size].reshape(-1))

	list_images = np.array(list_images).reshape((len(list_images),size**2))
	list_label_images = np.array(list_label_images).reshape((len(list_images)))
	list_images = list_images/255

	return list_images, list_label_images


def load_B(size = 128.):
	dir = "./images/OnlineDatabases/DatasetB/"

	list_label_images_raw = os.listdir(dir + "labels_two/")

	list_label_images= [] 
	list_images = []
	for i, j in enumerate(list_label_images_raw):
		if (j[0:3] =="two" and  j[-22:-21] == "R"):
			file = str(j[8:-18] + ".tif")
			temp = np.array(Image.open(str(dir + "labels_two/"+ j)))
			temp2 = np.array(Image.open(str(dir + file)))
		
			
			for i in range(int(float(temp.shape[0])/float(size))):
				for j in range(int(float(temp.shape[1])/float(size))):
					
					list_label_images.append(np.sum(temp[i:i+size,j:j+size]/(255 * size**2)))
					list_images.append(temp2[i:i+size,j:j+size].reshape(-1))

	list_images = np.array(list_images).reshape((len(list_images),size**2))
	list_label_images = np.array(list_label_images).reshape((len(list_images)))

	list_images = list_images/255
	return list_images, list_label_images

			
					
					









