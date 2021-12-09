import tensorflow as tf
import numpy as np
import math


class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, imgs, kps, SIGMA, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_KEYPOINTS, shuffle=True):
        self.imgs = imgs
        self.kps = kps
        self.batch_size = BATCH_SIZE
        self.sigma = SIGMA
        self.img_w = IMG_WIDTH
        self.img_h = IMG_HEIGHT
        self.num_kp = NUM_KEYPOINTS
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.imgs) // self.batch_size

    def __getitem__(self, index):
        #Get index of images to generate
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        #Shuffle the data after the generator has run through all samples
        self.indexes = np.arange(len(self.imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)                
            
    def gaussian(self, xL, yL, H, W, sigma=5):
        ##Function that creates the heatmaps##
        channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
        channel = np.array(channel, dtype=np.float32)
        channel = np.reshape(channel, newshape=(H, W))

        return channel

    def __data_generation(self, indexes):
        #Generates data containing batch_size samples
        X_batch = [self.imgs[i] for i in indexes]
        X_batch = np.array(X_batch)
            
        y_batch = []
        
        kps = [self.kps[i] for i in indexes]
        
        for i in range(0,len(kps)):
            heatmaps = []
            for j in range(0, self.num_kp):
                if str(kps[i][j*2]) != 'nan':
                    x = int(kps[i][j*2])
                    y = int(kps[i][j*2+1])
                    heatmap = self.gaussian(x, y, self.img_h, self.img_w)
                else:
                    heatmap = np.zeros([self.img_h, self.img_w], dtype=np.float32)
                heatmaps.append(heatmap)
            y_batch.append(heatmaps)
                
        y_batch = np.array(y_batch)
        y_batch = np.swapaxes(y_batch,1,3)
        y_batch = np.swapaxes(y_batch,1,2)
        
        return X_batch, [y_batch, y_batch]
