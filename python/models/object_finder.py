import cv2
import pdb
import math
import time
import os.path
import numpy as np
from os import path
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense

class object_finder:
    def __init__(self, load_model_path=None, save_model_path=None, new_data=False, train=False):
        if train:
            if new_data:
                self.new_labels()
            self.init_fish_data()
            self.train_fish()
            
        self.model = None
        if load_model_path is not None and os.path.isfile(load_model_path):
            self.model = load_model(load_model_path, compile=False)
        
        self.fish_block_size = 27
        self.bar_block_size = 158
        self.bar_max_top = 385
        self.bar_offset = 5
        self.last_bar_data = None
        
    def new_labels(self):
        fish_data_path = 'data\\fish_labels.txt'
        fish_labels_path = 'numpy_data\\fish_labels.npy'
        label_data = [list(map(int, line.rstrip('\n').split(',')[1:])) for line in open(fish_data_path, 'r')]
        label_data = np.array(label_data)
        np.save(fish_labels_path, label_data)

        
    def init_fish_data(self):
        train_image_path = 'numpy_data\\train_imgs.npy'
        fish_labels_path = 'numpy_data\\fish_labels.npy'
        train_data = np.load(train_image_path)
        labels_data = np.load(fish_labels_path)        
        predictions = np.array([cv2.imread('data\\171.jpg')])
        rows = train_data.shape[1]
        
        return train_data[:labels_data.shape[0]], labels_data, predictions, rows


    def train_fish(self):
        flag = 3
        X, y, predictions, rows = self.init_fish_data()
        X, y, predictions = self.reshape_data(X, y, predictions, rows, flag=flag, block_size=27)
        
        new_model = Sequential()
        sgd = optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=False)
        new_model.add(Dense(600, input_shape=(X.shape[1],), activation='sigmoid'))
        new_model.add(Dense(1, activation='sigmoid'))
        new_model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])
        new_model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        if self.model_save_path is not None:
            new_model.save(self.model_save_path)

        
    def locate_fish(self, screen):
        screen = np.mean(screen, axis=2)
        new_x = list()
        for row in range(screen.shape[0] - self.fish_block_size):
            temp = screen[row:row+self.fish_block_size]
            new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))

        screen = np.array(new_x)
        fish_row = self.model.predict(screen)
        return np.argmax(fish_row)


    def locate_bar(self, screen):
        edges = cv2.Canny(screen, 100, 200)       
        edges  = edges / 255
        col_start = 10
        col_end = 30
        row_start = 12
        mask = np.ones(col_end-col_start)
        row = np.where((edges[row_start:,col_start:col_end]==mask).all(axis=1))
        if row[0].size > 0:
            row = row[0][0]
        else:
            if self.last_bar_data is not None:
                return self.last_bar_data
            return (0, 0)
        
        if row > self.bar_max_top:
            if self.last_bar_data is not None and abs(self.last_bar_data[0] - (row-self.bar_block_size+self.bar_offset)) > 130:
                return self.last_bar_data

            self.last_bar_data = (row-self.bar_block_size+self.bar_offset), row+self.bar_offset
            return (row-self.bar_block_size+self.bar_offset), row+self.bar_offset
        
        #Found bar top
        if row != 0:
            if self.last_bar_data is not None and abs(self.last_bar_data[0] - (row+self.bar_offset)) > 130:
                return self.last_bar_data
            
            self.last_bar_data = (row+self.bar_offset, row+self.bar_block_size+self.bar_offset)
            return row+self.bar_offset, row+self.bar_block_size+self.bar_offset

            if self.last_bar_data is not None:
                return self.last_bar_data
        
        return self.last_bar_data

    
    def reshape_data(self, X, y, predictions, rows, flag=0, block_size=0):
        """
        Flag:
            0 (default) : shape as just matching 1 row (1 output layer)
            1 : Match every row of fish (27 output layers)
            2 : Reshape input to be n inputs of 27, 1 output to see if yes or no fish
            3 : do same as 2, but average the RGB values for each pixel
        """

        if flag == 0:
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
            #Reshape from from just a # to all 0's
            reshaped_target = np.zeros((y.shape[0], rows))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y]=1

            return X, reshaped_target, predictions

        
        elif flag == 1:
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))

            predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
        
            #Reshape from from just a # to all 0's
            reshaped_target = np.zeros((y.shape[0], rows))

            #Change index of correct predictions to a 1
            for row in range(reshaped_target.shape[0]):
                reshaped_target[row,y[row][0]:y[row][1]]=1

            #pdb.set_trace()
            return X, reshaped_target, predictions

        
        elif flag == 2:
            in_range = block_size

            #Reshape inputs
            new_x = list()
            for inp in X:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_x.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
                          
            new_x = np.array(new_x)
            X = new_x

            #Reshape target
            reshaped_target = np.zeros((y[:,0].shape[0], rows - block_size))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
            y = reshaped_target
            y = y.reshape(y.shape[0] * y.shape[1])

            #Reshape predictions
            new_preds = list()
            for inp in predictions:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
            new_preds = np.array(new_preds)
            predictions = new_preds
            return X, y, predictions

        elif flag == 3:
            in_range = block_size
            X = np.mean(X, axis=3)
            #Reshape inputs
            new_x = list()
            for inp in X:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))
                          
            new_x = np.array(new_x)
            
            X = new_x

            #Reshape target
            reshaped_target = np.zeros((y[:,0].shape[0], rows - block_size))

            #Change index of correct predictions to a 1
            reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
            y = reshaped_target
            y = y.reshape(y.shape[0] * y.shape[1])

            #Reshape predictions
            predictions = np.mean(predictions, axis=3)
            new_preds = list()
            for inp in predictions:
                for row in range(inp.shape[0] - block_size):            
                    temp = inp[row:row+in_range]
                    new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1]))
            new_preds = np.array(new_preds)
            predictions = new_preds
            return X, y, predictions



if __name__ == '__main__':
    ol = object_finder(load_model_path='batch100_fish_id.h5')
    train_data, _, _, _ = ol.init_fish_data()
    #cv2.imwrite('image_data\\edges61.jpg', cv2.Canny(train_data[60], 100, 200))
    #cv2.imwrite('image_data\\edges62.jpg', cv2.Canny(train_data[61], 100, 200))
    #cv2.imwrite('image_data\\edges63.jpg', cv2.Canny(train_data[62], 100, 200))
    bottom = 0
    top = 134
    print(ol.locate_fish(train_data[bottom]))
    print(ol.locate_bar(train_data[top]))
    #for i in range(29, 40):
    #print(str(i+1) + ": " + str(ol.locate_bar(train_data[i])))
    #pdb.set_trace()
    #for i in range(0, 10):
    #    print(locate_bar(train_data[i]))
    #train_fish()
    #model = load_model('batch100_fish_id.h5', compile=False)
    #print(locate_fish(cv2.imread('data\\1.jpg')))
    #train_bar()
