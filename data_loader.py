from os import listdir, system, mkdir
from os.path import isfile, join
import os
import cv2
import numpy as np
from tqdm import tqdm
import h5py
from glob import glob


def download_dataset():
	''' Download and extract dataset. '''
	system('wget http://vis-www.cs.umass.edu/lfw/lfw.tgz')
	system('tar -xzvf lfw.tgz')
	system('rm lfw.tgz')
	system('mv lfw facecolor')

def split_dataset():
	''' Split dataset into trainA and testA. '''
	dirlist = listdir('facecolor')
	i = 1
	for di in dirlist:
		system("mv facecolor/"+di+"/* facecolor")
		system("rmdir facecolor/"+di)
		print(str(i)+"/"+str(len(dirlist)),end='\r')
		i+=1

	flist = listdir('facecolor')

	train = ['facecolor/'+i for i in flist[:int(0.8*len(flist))]]

	# Taking only a subset because insufficient memory
	# Chunking because mv cannot take whole list at once
	train1 = train[:3000]
	train1_str = " ".join(train1)

	train2 = train[3000:5000]
	train2_str = " ".join(train2)

	test = ['facecolor/'+i for i in flist[int(0.8*len(flist)):]][:1000]
	test_str = " ".join(test)

	mkdir('facecolor/trainA')
	system("mv "+train1_str+" facecolor/trainA")
	system("mv "+train2_str+" facecolor/trainA")

	mkdir('facecolor/testA')
	system("mv "+test_str+" facecolor/testA")

	# remove unneeded images
	system("rm facecolor/*jpg")


def create_B_dataset():
	''' Create black and white dataset i.e. trainB and testB. '''
	TRAIN_PATH = 'facecolor/trainA'
	TEST_PATH = 'facecolor/testA'

	files = {TRAIN_PATH:[f for f in listdir(TRAIN_PATH)],
	     TEST_PATH:[f for f in listdir(TEST_PATH)]}

	SAVE_TRAIN_PATH = 'facecolor/trainB'
	SAVE_TEST_PATH = 'facecolor/testB'
	mkdir(SAVE_TRAIN_PATH)
	mkdir(SAVE_TEST_PATH)

	print('Creating black and white version of dataset')
	COUNT = 0
	for k,v in files.items():
		if k == TRAIN_PATH:
			savepath = SAVE_TRAIN_PATH
		else:
			savepath = SAVE_TEST_PATH
		for i in v:
			path = k + '/' + i
			im1 = cv2.imread(path)
			im1 = cv2.resize(im1,(256,256))
			cv2.imwrite(path,im1)

			im2 = cv2.imread(path,0)
			cv2.imwrite(savepath + '/' + i,im2)

			COUNT+=1

def read_image(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	if len(image.shape) != 3 or image.shape[2] != 3:
		print('Wrong image {} with shape {}'.format(path, image.shape))
		return None

	# range of pixel values = [-1.0, 1.0]
	image = image.astype(np.float32) / 255.0
	image = image * 2.0 - 1.0
	return image

def read_images():
	ret = []
	base_dir = 'facecolor'
	for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
		data_dir = os.path.join(base_dir, dir_name)
		paths = glob(os.path.join(data_dir, '*.jpg'))
		print('# images in {}: {}'.format(data_dir, len(paths)))

		images = []
		for path in tqdm(paths):
			image = read_image(path)
			if image is not None:
				images.append(image)
		ret.append((dir_name, images))
	return ret

def store_h5py(dir_name, images, image_size):
	base_dir = 'facecolor'
	f = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'w')
	for i in range(len(images)):
		grp = f.create_group(str(i))
		if images[i].shape[0] != image_size:
			image = imresize(images[i], (image_size, image_size, 3))
			# range of pixel values = [-1.0, 1.0]
			image = image.astype(np.float32) / 255.0
			image = image * 2.0 - 1.0
			grp['image'] = image
		else:
			grp['image'] = images[i]
	f.close()

def read_h5py():
	base_dir = 'facecolor/'
	image_size = 256
	paths = glob(os.path.join(base_dir, '*_{}.hy'.format(image_size)))
	if len(paths) != 4:
		convert_h5py()
	ret = []
	for dir_name in ['trainA', 'trainB', 'testA', 'testB']:
		try:
			dataset = h5py.File(os.path.join(base_dir, '{}_{}.hy'.format(dir_name, image_size)), 'r')
		except:
			raise IOError('Dataset is not available. Please try it again')
		images = []
		for id in dataset:
			images.append(dataset[id]['image'].value.astype(np.float32))
		ret.append(images)
	return ret

def convert_h5py():
	print('Generating h5py file')
	data = read_images()
	for dir_name, images in data:
		store_h5py(dir_name, images, 256)

def get_data():
	convert_h5py()

	print('Load data:')
	train_A, train_B, test_A, test_B = read_h5py()
	return train_A, train_B, test_A, test_B

if __name__ == "__main__":
	download_dataset()
	split_dataset()
	create_B_dataset()
	convert_h5py()
