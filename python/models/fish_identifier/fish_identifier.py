import cv2
import pdb
import time
import os.path
import numpy as np
from os import path
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense


def init_data():
    #new_labels()
    #exit
    train_data = np.load('train_imgs.npy')
    labels_data = np.load('fish_labels.npy',)
    #edges = cv2.Canny(train_data[7], 200, 300)
    #pdb.set_trace()
    #62,64
    #pdb.set_trace()
    #cv2.imshow('', edges)
    
    predictions = np.array([cv2.imread('data\\171.jpg'),
                        cv2.imread('data\\172.jpg'),
                        cv2.imread('data\\173.jpg'),
                        cv2.imread('data\\174.jpg'),
                        cv2.imread('data\\175.jpg'),
                        cv2.imread('data\\176.jpg'),
                        cv2.imread('data\\177.jpg'),
                        cv2.imread('data\\178.jpg'),
                        cv2.imread('data\\179.jpg'),
                        cv2.imread('data\\180.jpg'),
                        cv2.imread('data\\181.jpg'),
                        cv2.imread('data\\182.jpg'),
                        cv2.imread('data\\183.jpg'),
                        cv2.imread('data\\184.jpg'),
                        cv2.imread('data\\185.jpg'),
                        cv2.imread('data\\186.jpg'),
                        cv2.imread('data\\187.jpg'),
                        cv2.imread('data\\188.jpg'),
                        cv2.imread('data\\189.jpg'),
                        cv2.imread('data\\190.jpg'),
                        cv2.imread('data\\191.jpg'),
                        cv2.imread('data\\192.jpg'),
                        cv2.imread('data\\193.jpg'),
                        cv2.imread('data\\194.jpg'),
                        cv2.imread('data\\195.jpg'),
                        cv2.imread('data\\196.jpg'),
                        cv2.imread('data\\197.jpg'),
                        cv2.imread('data\\198.jpg'),
                        cv2.imread('data\\199.jpg'),
                        cv2.imread('data\\200.jpg')])

    rows = train_data.shape[1]    
    return train_data[:170], labels_data, predictions, rows



def new_labels():
    #pdb.set_trace()
    label_data = [list(map(int, line.rstrip('\n').split(',')[1:])) for line in open('data/fish_labels.txt', 'r')]
    label_data = np.array(label_data)
    np.save('fish_labels.npy', label_data)


def locate_fish(screen):
    in_range = 27
    screen = np.mean(screen, axis=2)
    #Reshape inputs
    new_x = list()
    for row in range(screen.shape[0] - 27):
        temp = screen[row:row+in_range]
        new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))

    #pdb.set_trace()  
    screen = np.array(new_x)
    fish_row = model.predict(screen)
    return np.argmax(fish_row)


def train():
    flag = 3
    X, y, predictions, rows = init_data()
    X, y, predictions = reshape_data(X, y, predictions, rows, flag=flag)
    
    model = Sequential()
    if not path.exists('fish_id.h5'):
        sgd = optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=False)
        model.add(Dense(600, input_shape=(X.shape[1],), activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='Adadelta', metrics=['accuracy'])
        model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        #model.save('fish_id.h5')
    else:
        model = load_model('fish_id.h5')

    #results = model.evaluate(Xtest, ytest, batch_size=5)
    #print('test loss, test acc:', results)

    start = time.time()
    preds = model.predict(predictions)
    if flag == 2 or flag == 3:
        preds = preds.reshape(int(preds.shape[0] / 523), 523)
    print(np.argmax(preds, axis=1))
    print(time.time() - start)
    pdb.set_trace()
    
def reshape_data(X, y, predictions, rows, flag=0):
    """
    Flag:
        0 (default) : shape as just matching 1 row (1 output layer)
        1 : Match every row of fish (27 output layers)
        2 : Reshape input to be n inputs of 27, 1 output to see if yes or no fish
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
        in_range = 27

        #Reshape inputs
        new_x = list()
        for inp in X:
            for row in range(inp.shape[0] - 27):            
                temp = inp[row:row+in_range]
                new_x.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
                      
        new_x = np.array(new_x)
        X = new_x

        #Reshape target
        reshaped_target = np.zeros((y[:,0].shape[0], rows - 27))

        #Change index of correct predictions to a 1
        reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
        y = reshaped_target
        y = y.reshape(y.shape[0] * y.shape[1])

        #Reshape predictions
        new_preds = list()
        for inp in predictions:
            for row in range(inp.shape[0] - 27):            
                temp = inp[row:row+in_range]
                new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
        new_preds = np.array(new_preds)
        predictions = new_preds
        return X, y, predictions

    elif flag == 3:
        in_range = 27
        X = np.mean(X, axis=3)
        #Reshape inputs
        new_x = list()
        for inp in X:
            for row in range(inp.shape[0] - 27):            
                temp = inp[row:row+in_range]
                new_x.append(temp.reshape(temp.shape[0] * temp.shape[1]))
                      
        new_x = np.array(new_x)
        
        X = new_x

        #Reshape target
        reshaped_target = np.zeros((y[:,0].shape[0], rows - 27))

        #Change index of correct predictions to a 1
        reshaped_target[np.arange(reshaped_target.shape[0]), y[:,0]]=1
        y = reshaped_target
        y = y.reshape(y.shape[0] * y.shape[1])

        #Reshape predictions
        predictions = np.mean(predictions, axis=3)
        new_preds = list()
        for inp in predictions:
            for row in range(inp.shape[0] - 27):            
                temp = inp[row:row+in_range]
                new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1]))
        new_preds = np.array(new_preds)
        predictions = new_preds
        return X, y, predictions
#train()
model = load_model('batch100_fish_id.h5', compile=False)
