import os
import sys

FLAGS = None

up1 = os.path.abspath('../../utils/') 
up2 = os.path.abspath('../../models/') 
sys.path.insert(0, up1)
sys.path.insert(0, up2)

from optical_flow_data_gen import DataGenerator
from ucf101_data_utils import get_test_data_opt_flow, get_train_data_opt_flow
from motion_network import getKerasCifarMotionModel2, getKerasCifarMotionModelOnly
from keras.optimizers import SGD
from matplotlib import pyplot as plt
from keras.optimizers import SGD
import cv2 
import numpy as np
import keras
import datetime
import argparse
import pickle

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.initializers import Ones
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback

def getModel(lr=1e-2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 10)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))

    optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def get_callbacks(filepath):
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [msave]

from optical_flow_data_gen import DataGenerator
from ucf101_data_utils import get_test_data_opt_flow, get_train_data_opt_flow

training_options = { 'rescale' : 1./255, 
                 'samplewise_center': True, 
                     'samplewise_std_normalization':True,
                        'zoom_range' : 0.2, 
                        'horizontal_flip' : True}

validation_options = { 'rescale' : 1./255,
                 'samplewise_center': True, 
                     'samplewise_std_normalization':True}




params_train = { 'data_dir' : "/data/tvl1_flow",
      'dim': (224,224),
      'batch_size': 64,
      'n_frames': 5,
      'n_frequency': 2,
      'shuffle': True, 
            'n_classes' : 10,
       'validation' : False,
            'enable_augmentation' : False,
           'training_opts' : training_options}

params_valid = { 'data_dir' : "/data/tvl1_flow",
      'dim': (224,224),
      'batch_size':64,
      'n_frames': 5,
      'n_frequency': 2,
      'shuffle': True, 
            'n_classes' : 10,
       'validation' : True,
           'validation_opts' : validation_options}
    

def run_model(model_load_path, model_save_file, num_epochs, learning_rate, history_pickle_file):
    

    id_labels_train = get_train_data_opt_flow('../../data/ucf101_splits/trainlist01_small.txt')
    labels = id_labels_train[1]
    id_test = get_test_data_opt_flow('../../data/ucf101_splits/testlist01_small.txt', \
                           '../../data/ucf101_splits/classInd_small.txt')

    training_generator = DataGenerator(*id_labels_train, **params_train)


    validation_generator = DataGenerator(id_test[0], id_test[1], **params_valid)
    
    
    callbacks = get_callbacks(filepath=model_save_file)

    model_slow_lr = getModel(lr=learning_rate)
    
    if model_load_path != '':
        model_slow_lr.load_weights(model_load_path)
           
    mod1 = model_slow_lr.fit_generator(generator=training_generator, steps_per_epoch=32,
                    validation_data=validation_generator, validation_steps=32,
                    use_multiprocessing=True,
                    workers=1, epochs=num_epochs,
                    verbose=1, callbacks=callbacks)
    
    with open(history_pickle_file, 'wb') as handle:
        pickle.dump(mod1.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', type=str, default='experiment',
                      help='experiment name')
        
    parser.add_argument('--model_save_file', type=str,
                      default='./model.hdf5',
                      help='file to store model weights')
    
    parser.add_argument('--model_load_path', type=str,
                      default='',
                      help='model to load on start up')
    
    parser.add_argument('--log_file', type=str, 
                        default='./log' + datetime.datetime.now().isoformat() \
                        + '__x' + '.txt', help='Log file')
    
    parser.add_argument('--num_epochs', type=int, 
                        default=1, help='number of epochs to run training')
    
    parser.add_argument('--history_pickle_file', type=str, 
                        default='./history.pickle', help='loss/accuracy file store')
    
    parser.add_argument('--learning_rate', type=float, 
                        default=1e-2, help='learning rate')
    
    FLAGS, unparsed = parser.parse_known_args()
   
    log_file = str(FLAGS.log_file).replace('_x', FLAGS.experiment_name)
        
    file = open(log_file, 'w')
    sys.stdout = file 
    
    print('model save file: ', FLAGS.model_save_file)
    print('model load path', FLAGS.model_load_path)
    print('log file: ', log_file)
    print('experiment name: ', FLAGS.experiment_name)
    print('num epochs: ', FLAGS.num_epochs)
    print('Learning rate: ', FLAGS.learning_rate)
    print('history pickle file: ', FLAGS.history_pickle_file)
    
    run_model(FLAGS.model_load_path, FLAGS.model_save_file, FLAGS.num_epochs, \
             FLAGS.learning_rate, FLAGS.history_pickle_file)
    
    print("finished!")
