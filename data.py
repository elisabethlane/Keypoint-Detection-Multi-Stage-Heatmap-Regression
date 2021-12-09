import pandas as pd
import numpy as np
from PIL import Image


class DataSet():
    def __init__(self, IMG_HEIGHT, IMG_WIDTH):
        self.img_w = IMG_WIDTH
        self.img_h = IMG_HEIGHT
        self.train_filenames = self.get_train_filenames()
        self.test_filenames = self.get_test_filenames()
        self.train_image_data = self.get_train_images()
        self.train_label_data = self.get_train_labels()
        self.test_image_data = self.get_test_images()
        self.test_label_data = self.get_test_labels()
        self.num_train_ims = len(self.train_filenames)
        self.num_test_ims = len(self.test_filenames)
        
    def get_test_filenames(self):
        data = pd.read_csv("data/test_labels.csv", header=None, dtype="str").drop([1, 2, 3, 4],axis=1).values.tolist()
        return data
    
    def get_train_filenames(self):
        data = pd.read_csv("data/train_labels.csv", header=None, dtype="str").drop([1, 2, 3, 4],axis=1).values.tolist()
        return data

    def get_train_images(self):
        
        image_sequence = []
        
        for i in range(0, len(self.train_filenames)):
            img_name = self.train_filenames[i][0]
            image = Image.open(f"data/images/train/{img_name}")
            image = np.asarray(image)
            image = (image / 255.)
            image_sequence.append(image)
            
        image_list = np.array(image_sequence, dtype = "float")
        
        X_train = image_list.reshape(-1, self.img_h, self.img_w, 1)
        
        return X_train
    
    def get_test_images(self):
        
        image_sequence = []
        
        for i in range(0, len(self.test_filenames)):
            img_name = self.test_filenames[i][0]
            image = Image.open(f"data/images/test/{img_name}")
            image = np.asarray(image)
            image = (image / 255.)
            image_sequence.append(image)
            
        image_list = np.array(image_sequence, dtype = "float")
        
        X_test = image_list.reshape(-1, self.img_h, self.img_w, 1)
        
        return X_test
    
    
    def get_train_labels(self):
        
        data = pd.read_csv("data/train_labels.csv", header=None, dtype="str").drop([0],axis=1).values.tolist()
        y_train = np.array(data, dtype = 'float')
        
        return y_train
    
    def get_test_labels(self):
        
        data = pd.read_csv("data/test_labels.csv", header=None, dtype="str").drop([0],axis=1).values.tolist()
        y_test = np.array(data, dtype = 'float')
        
        return y_test


