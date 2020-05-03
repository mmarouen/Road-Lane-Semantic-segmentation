import cv2
import subprocess
import numpy as np
import random
from glob import glob
import os
import re

class DataUtils():
  def __init__(self, data_url, weights_url):
    self.data_url = data_url
    self.weights_url = weights_url
  
  def get(self, data_folder, n_classes, input_height, input_width):
    script_path = os.path.join(os.getcwd(),"load_from_gdrive.sh")
    subprocess.call("sudo %s %s %s"%(script_path, self.data_url, self.weights_url), shell=True)
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    X = []
    Y = []
    for image in image_paths:
      im = cv2.imread(image, 1)
      im = cv2.resize(im,(input_width, input_height))
      im = im.astype(np.float32)
      im = im/255.0
      X.append(im)
      
      seg_labels = np.zeros((input_height, input_width, n_classes))
      segpath = label_paths[os.path.basename(image)]
      seg = cv2.imread(segpath, 1)
      seg = cv2.resize(seg,(input_width, input_height))
      seg = seg[:, : , 0]
      for c in range(n_classes):
        if c==0:
          seg_labels[: , : , c ] = (seg == c).astype(int)
        else:
          seg_labels[: , : , c ] = (seg > 0 ).astype(int)
      seg_labels = np.reshape(seg_labels, (input_height*input_width, n_classes))
      Y.append(seg_labels)
    return np.asarray(X), np.asarray(Y)

  def augment_training_data(self, train_im,train_lab,n_classes,input_height,input_width):
    augmented_lab=[]
    augmented_im=[]
    for i in range(len(train_im)):
      im=train_im[i,:,:,:].reshape(input_height, input_width, 3)
      augmented_im.append(im)
      augmented_im.append(cv2.flip(im,1))
      lab=train_lab[i,:,:].reshape(input_height,input_width,n_classes)
      augmented_lab.append(np.reshape(lab,(input_height*input_width,n_classes)))
      augmented_lab.append(np.reshape(cv2.flip(lab,1),(input_height*input_width,n_classes)))
    l = list(zip(augmented_lab,augmented_im))
    random.shuffle(l)
    augmented_lab,augmented_im=zip(*l)
    augmented_im=np.asarray(augmented_im)
    augmented_lab=np.asarray(augmented_lab)
    return augmented_im, augmented_lab
