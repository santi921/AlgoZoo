
####Template for future datasets
"""

def load_C(size = 128, tf_aug = False):
	stats = LABEL DIRECTORY/DATABASE
	dir_C = IMAGE DIRECTORY 

	list_label_images_raw = os.listdir(dir_A)
	list_label_images= [] 
	list_images = []

	for index, row in stats.iterrows():

		scale_temp = row["scale_val"]
		temp_image = np.array(Image.open(dir_A + row["id"] + "standard.png")).repeat(scale_temp,axis=0).repeat(scale_temp,axis=1)

		temp_image = gaussian_filter(temp_image, 6)

		for i in range(int(float(temp_image.shape[0])/float(size))):
				for j in range(int(float(temp_image.shape[1])/float(size))):
				
					list_label_images.append(row["labels"]) #ASSUMES DATABASE HAS A LABEL CATEGORY TO LOAD

					if (tf_aug == False ): 
						list_images.append(temp_image[i:i+size,j:j+size].reshape(-1))
					else: 
						list_images.append(temp_image[i:i+size,j:j+size])

	if (tf_aug == False):
		list_images = np.array(list_images).reshape((len(list_images),size**2))
	
	list_label_images = np.array(list_label_images).reshape((len(list_images)))
	list_images = np.array(list_images)/255

	return list_images, list_label_images

"""