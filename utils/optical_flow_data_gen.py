from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras
import cv2

import os, os.path
import random

default_validation_options = { 'rescale' : 1./255, \
                         'samplewise_center': True, \
                         'samplewise_std_normalization':True \
    }

default_training_options = { 'rescale' : 1./255, \
                            'shear_range' : 0.2, \
                            'zoom_range' : 0.2, \
                         'samplewise_center' : True, \
                         'samplewise_std_normalization' : True \
    }

class DataGenerator(keras.utils.Sequence):
    'Generate UCF 101 data for keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=64, dim=(224,224), n_frames=3, n_frequency=5, n_classes=101, shuffle=True, \
                validation=False, return_files=False, enable_augmentation=False, feature_wise_standardization=False, \
                 training_opts=default_training_options, \
                 validation_opts=default_validation_options):        
        'Initialisation'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_frames = n_frames
        self.n_frequency = n_frequency
        self.shuffle = shuffle 
        self.n_classes = n_classes
        self.return_files = return_files
        self.enable_augmentation = enable_augmentation
        self.feature_wise_standardization = feature_wise_standardization
        self.mean = None
        self.std = None
        self.on_epoch_end()

        
        if validation :
            self.data_gen = ImageDataGenerator(**validation_opts)
        else :
            self.data_gen = ImageDataGenerator(**training_opts)
    
    def  __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def fit(self):
        X = np.empty((self.batch_size*32, *self.dim,self.n_frames*2), dtype=np.float64)        
        
        for i in range(0, 32):
            _X, _y = self.__getitem2__(i, False)
            X[i*self.batch_size:(i+1)*self.batch_size,] = _X
        
        self.mean = np.mean(X, axis=(0, 1, 2))
        broadcast_shape = [1, 1, 2]
        self.mean = np.reshape(self.mean, broadcast_shape)
        
        self.std = np.std(X, axis=(0, 1, 2))
        broadcast_shape = [1, 1, 2]
        self.std = np.reshape(self.std, broadcast_shape)
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data 
        X, y, dirs = self.__data_generation(list_IDs_temp)
        
        if self.return_files:
            return X, y, dirs
        else:
            return X, y
    
    def __getitem2__(self, index, standardization_enabled):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data 
        X, y, dirs = self.__data_generation(list_IDs_temp, standardization_enabled)
        
        if self.return_files:
            return X, y, dirs
        else:
            return X, y
    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp, standardization_enabled=True):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim,self.n_frames*2), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        dirs = []
        
        # Generate data 
        for i, ID in enumerate(list_IDs_temp):
            data = self.__data_load(ID)            
            X[i,] = data[0] 
            y[i] = self.labels[ID]-1
            dirs.append(data[1])
        
        if standardization_enabled:
            X = self.data_gen.standardize(X.astype(float))
            if self.feature_wise_standardization:
                if (self.mean is not None) and (self.std is not None):
                    X -= self.mean
                    X /= (self.std + np.finfo(float).eps)
        
        if self.enable_augmentation and self.n_frames*2 == 2:
            for i in range(0, X.shape[0]):
                X[i,:,:,:] = self.data_gen.random_transform(X[i,:,:,:])
                
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes), dirs
    
    def __data_load(self, ID):
        u_file_dir = self.data_dir + "/u/" + ID.rstrip('.avi')
        v_file_dir = self.data_dir + "/v/" + ID.rstrip('.avi')
        frame_count = len([name for name in os.listdir(u_file_dir) if name.endswith('.jpg')])
        
        # sequence length to sample fraome
        seq_len = self.n_frames * self.n_frequency 
                
        # select from a random sequence n frames long with a given sample rate 
        seq_start = random.randint(1, frame_count - seq_len + 1)
        
        if seq_len > frame_count:
            raise Exception("n_frame * n_frequency > frame_count")
        
        frames = [i for i in range(seq_start,seq_start+seq_len,self.n_frequency)]     
        img = None
        files = []
        for frame in frames:
            u_file = u_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
            v_file = v_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
            files.append(u_file)
            u_img = cv2.imread(u_file, 0)
            u_img = cv2.resize(u_img, self.dim) 
            u_img = u_img.reshape((*self.dim,1))
            v_img = cv2.imread(v_file, 0)
            v_img = cv2.resize(v_img, self.dim) 
            v_img = v_img.reshape((*self.dim,1))
            
            if img is None:
                img = np.concatenate((u_img, v_img), axis=2)
            else:
                img = np.concatenate((img, np.concatenate((u_img, v_img), axis=2)), axis=2)            
        return img, files
    
    def get_all_frames_random_video(self, n_sample):
        'select all frames for a randomly selected video' 
        
        index = random.randint(0,31)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        ID = list_IDs_temp[0]
        X = np.empty((n_sample, *self.dim,self.n_frames*2), dtype=np.float32)        
        y = self.labels[ID]-1
        
        # Generate data 
        for i in range(0, n_sample):
            X[i,] = self.__data_load(ID)
            
        return self.data_gen.standardize(X.astype(float)), y
    
    def get_all_frames_for_video_id(self, ID, n_sample):
        'select all frames for a video ID' 
        
        X = np.empty((n_sample, *self.dim,self.n_frames*2), dtype=np.float32)        
        y = self.labels[ID]-1
        
        # Generate data 
        for i in range(0, n_sample):
            X[i,] = self.__data_load(ID)
            
        return self.data_gen.standardize(X.astype(float)), y
    

class DataGeneratorRGB(keras.utils.Sequence):
    'Generate UCF 101 data for keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=64, dim=(224,224), n_classes=101, shuffle=True, \
                validation=False, return_files=False, training_opts=default_training_options, \
                 validation_opts=default_validation_options):        
        'Initialisation'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle 
        self.n_classes = n_classes
        self.return_files = return_files
        self.on_epoch_end()

        
        if validation :
            self.data_gen = ImageDataGenerator(**validation_opts)
        else :
            self.data_gen = ImageDataGenerator(**training_opts)
    
    def  __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data 
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim,3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data 
        for i, ID in enumerate(list_IDs_temp):   
            X[i,] = self.__data_load(ID)        
            y[i] = self.labels[ID]-1
            
        return self.data_gen.standardize(X.astype(float)), keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __data_load(self, ID):
        u_file_dir = self.data_dir + "/u/" + ID.rstrip('.avi')
        v_file_dir = self.data_dir + "/v/" + ID.rstrip('.avi')
        frame_count = len([name for name in os.listdir(u_file_dir) if name.endswith('.jpg')])        
                
        # select from a random sequence n frames long with a given sample rate 
        frame = random.randint(1, frame_count-1)        

        u_file = u_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
        v_file = v_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
        
        u_img = cv2.imread(u_file, 0)
        u_img = cv2.resize(u_img, self.dim) 
        u_img = u_img.reshape((*self.dim,1))
        v_img = cv2.imread(v_file, 0)
        v_img = cv2.resize(v_img, self.dim) 
        v_img = v_img.reshape((*self.dim,1))
           
        img = np.concatenate((u_img, v_img), axis=2)
        img = np.concatenate((img, u_img), axis=2)            
        return img
    
    
class DataGeneratorVideoAccuracy(keras.utils.Sequence):
    'Generate UCF 101 data for keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=1, dim=(224,224), n_classes=101, \
                 sample_size=25, validation=False, enable_augmentation=False, training_opts=default_training_options, \
                 validation_opts=default_validation_options):        
        'Initialisation'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.enable_augmentation = enable_augmentation
        self.sample_size = sample_size
        
        if validation :
            self.data_gen = ImageDataGenerator(**validation_opts)
        else :
            self.data_gen = ImageDataGenerator(**training_opts)
    
    def  __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate data 
        X, y = self.__data_generation(self.list_IDs[index])
        
        return X, y
    
    def on_epoch_end(self):
        'Update indexes after each epoch'
        pass
    
    def __data_generation(self, ID):
        'Generates data containing batch_size samples' 
        X = np.empty((self.sample_size, *self.dim,2), dtype=np.float32)
        y = np.empty((self.sample_size), dtype=int)
        
        print('ID: ', ID)
        # Generate data 
        data = self.__data_load(ID)  
        for i, image in enumerate(data):
            X[i,] = data[i] 
            y[i] = self.labels[ID]-1
        
        X = self.data_gen.standardize(X.astype(float))
        
        if self.enable_augmentation:
            X[:,:,:,0] = self.data_gen.random_transform(X[:,:,:,0])
            X[:,:,:,1] = self.data_gen.random_transform(X[:,:,:,1])
        
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __data_load(self, ID):
        u_file_dir = self.data_dir + "/u/" + ID.rstrip('.avi')
        v_file_dir = self.data_dir + "/v/" + ID.rstrip('.avi')
        
        frame_count = len([name for name in os.listdir(u_file_dir) if  name.endswith('.jpg')])
        frame_range = range(1,frame_count+1,1)
        
        # select num_frames random frames
        frames = random.sample(frame_range, self.sample_size)
        img = None
        images = []
        for frame in frames:
            u_file = u_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
            v_file = v_file_dir + os.sep + 'frame' + f'{frame:06}' + '.jpg'
            u_img = cv2.imread(u_file, 0)
            u_img = cv2.resize(u_img, self.dim) 
            u_img = u_img.reshape((*self.dim,1))
            v_img = cv2.imread(v_file, 0)
            v_img = cv2.resize(v_img, self.dim) 
            v_img = v_img.reshape((*self.dim,1))            
            img = np.concatenate((u_img, v_img), axis=2)
            images.append(img)
        return images