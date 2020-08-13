import argparse
import warnings
with warnings.catch_warnings():
  warnings.filterwarnings("ignore",category=FutureWarning)
  import tensorflow as tf
from model import CycleGAN
import numpy as np
import cv2
import h5py
import os

parser = argparse.ArgumentParser(description='Face Colorization with a CycleGAN.')
parser.add_argument("--train", help="Train the network.", action='store_true')
parser.add_argument("--test-single", help="A single image to colorize", type=str)
args = parser.parse_args()

print('Build graph:')
model = CycleGAN()

init_all_op = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

if args.test_single is not None:
  with tf.compat.v1.Session() as sess:
          print("Initializing all parameters.")
          sess.run(init_all_op)

          saver.restore(sess, tf.train.latest_checkpoint('trained'))

          im = cv2.imread(args.test_single)
          print(im.shape)
          im = cv2.resize(im, (256, 256))
          model.test_single(sess,im)
else:
  ################################
  #          LOAD DATA           #
  ################################

  if not os.path.isdir("ckpt"):
          os.mkdir("ckpt")

  print("Load data:")
  trainAFile = h5py.File('facecolor/trainA_256.hy','r')
  trainA = []
  for i in list(trainAFile.keys()):
  	trainA.append(trainAFile[i]['image'])

  trainBFile = h5py.File('facecolor/trainB_256.hy','r')
  trainB = []
  for i in list(trainBFile.keys()):
  	trainB.append(trainBFile[i]['image'])

  testAFile = h5py.File('facecolor/testA_256.hy','r')
  testA = []
  for i in list(testAFile.keys()):
  	testA.append(testAFile[i]['image'])

  testBFile = h5py.File('facecolor/testB_256.hy','r')
  testB = []
  for i in list(testBFile.keys()):
  	testB.append(testBFile[i]['image'])


  ################################
  #           TRAINING           #
  ################################


  init_all_op = tf.compat.v1.global_variables_initializer()
  saver = tf.compat.v1.train.Saver()

  with tf.compat.v1.Session() as sess:
          print("Initializing all parameters.")
          sess.run(init_all_op)

          if tf.train.latest_checkpoint('ckpt') is not None:
            saver.restore(sess, tf.train.latest_checkpoint('ckpt'))

          print("Starting training session.")
          model.train(sess, saver, trainA, trainB)

          if not os.path.isdir("results"):
                  os.mkdir("results")

          model.test(sess,testA,testB,'results')

