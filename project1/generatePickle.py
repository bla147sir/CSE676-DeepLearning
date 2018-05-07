import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import misc
import scipy.io
import numpy as np
import glob
from datetime import datetime, timedelta
#import dill as pickle

# following should be deleted or commented afer trial
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

x, y = mnist.train.next_batch(1)
print(mnist.train.images.shape[0])
print(mnist.test.images.shape[0])
print(x.shape, 'type: ',type(x))
print(y.shape)
'''
def resize_img(img_path):
	img = scipy.ndimage.imread(img_path, flatten=True)
	img = misc.imresize(img, (100, 100))
	img = img.flatten()

	# img = np.array([img])

	return list(img)

def mat_loader(path):
	# matfile = '/home/jzhao/Desktop/wiki_crop/wiki.mat'
	mat = scipy.io.loadmat(path)

	wiki = mat['wiki']

	item = wiki[0][0]

	dob = item[0][0]
	photo_taken = item[1][0]
	photo_path = item[2][0]

	label_dict = {}

	for i in range(len(dob)):
		birth = datetime.fromordinal(int(dob[i]))+timedelta(days=int(dob[i]%1))-timedelta(days=366)
		label_dict[photo_path[i][0]] = photo_taken[i]-birth.year

	return label_dict

def generateData(mat_file, img_directory):
	images_and_labels = []

	mat_path = mat_file #'/home/jzhao/Desktop/wiki_crop/wiki.mat'
	label_dict = mat_loader(mat_path)
	print(len(label_dict))

	count=0
	for folder in range(5):  #!!! changed from 100 to 10
		path = img_directory #'/home/jzhao/Desktop/wiki_crop/'
		f_name = str(folder) if folder>=10 else '0'+str(folder)
		path += f_name + '/*.jpg'
		print('reading images in file:', f_name)
		for filename in glob.glob(path):
			img = scipy.ndimage.imread(filename)
			if img.shape[0]<=300 and img.shape[1]<=300:
				data_x = resize_img(filename)
				data_y = np.zeros(100)
				index = label_dict[f_name+'/'+filename.split('/')[-1]]
				# if index > len(data_y):
				index = int(len(data_y)/2) if index >= len(data_y) else index
				data_y[index] += 1 #one_hot
				data_y = list(data_y)
				count += 1

				# images_and_label = []
				# images_and_label.append(data_x)
				# images_and_label.append(data_y)
				# images_and_labels.append(images_and_label)
				images_and_labels.append([data_x, data_y])

	print('totally processed:', count, 'and', len(images_and_labels))
	test_size = int(len(images_and_labels)*0.1)

	# print(type(images_and_labels[0][0]))
	# print(len(images_and_labels[:][0]))
	# print(images_and_labels[0][1].shape)
	final = np.array(images_and_labels)
	print('extract from', final.shape)

	'''
	[
		[ [data],[label] ]
	]

	'''

	train_x = list(final[:, 0][:-test_size])
	train_y = list(final[:, 1][:-test_size])

	test_x = list(final[:, 0][-test_size:])
	test_y = list(final[:, 1][-test_size:])

	print(len(train_x), len(test_x))
	print(len(train_y), len(test_y))
	print(len(train_x[0]), len(train_y[0]))
	'''
	with open('facialimage_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)
	'''
	return train_x, train_y, test_x, test_y





