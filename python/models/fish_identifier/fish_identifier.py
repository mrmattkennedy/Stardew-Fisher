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
    train_data = np.load('train_imgs.npy')
    labels_data = np.load('labels.npy',)
    
    
    predictions = np.array([cv2.imread('data\\64.jpg'),
                            cv2.imread('data\\65.jpg'),
                            cv2.imread('data\\66.jpg'),
                            cv2.imread('data\\67.jpg'),
                            cv2.imread('data\\68.jpg'),])

    #pdb.set_trace()
    #label_data = [list(map(int, line.rstrip('\n').split(',')[1:])) for line in open('data/labels.txt', 'r')]
    #label_data = np.array(label_data)
    #np.save('labels.npy', label_data)
    rows = train_data.shape[1]
    #train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2] * train_data.shape[3]))
    #predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2] * predictions.shape[3]))
    
    return train_data[:63], labels_data, predictions, rows
    
def train():
    X, y, predictions, rows = init_data()
    in_range = 27
    new_x = list()
    for inp in X:
        for row in range(inp.shape[0] - 27):            
            temp = inp[row:row+in_range]
            new_x.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
                  
    new_x = np.array(new_x)
    X = new_x
    y = reshape_data(y[:,0], rows)
    y = y.reshape(y.shape[0] * y.shape[1])

    new_preds = list()
    for inp in predictions:
        for row in range(inp.shape[0] - 27):            
            temp = inp[row:row+in_range]
            new_preds.append(temp.reshape(temp.shape[0] * temp.shape[1] * temp.shape[2]))
    new_preds = np.array(new_preds)
    predictions = new_preds
    
    model = Sequential()
    if not path.exists('fish_id.h5'):
        sgd = optimizers.SGD(lr=0.001, decay=0.001, momentum=0.9, nesterov=False)
        model.add(Dense(400, input_shape=(X.shape[1],), activation='hard_sigmoid'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        model.fit(X, y, epochs=5, batch_size=X.shape[0], verbose=1)
        #model.save('fish_id.h5')
    else:
        model = load_model('fish_id.h5')

    #results = model.evaluate(Xtest, ytest, batch_size=5)
    #print('test loss, test acc:', results)

    for i in range(0, 5):
        start = time.time()
        preds = model.predict(predictions)
        preds = preds.reshape(int(preds.shape[0] / 523), 523)
        print(np.argmax(preds, axis=1))
        print(time.time() - start)
    pdb.set_trace()
    
def reshape_data(target, rows):
    """
    target_range = target[0][1] - target[0][0]
    
    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((target.shape[0], rows))

    #Change index of correct predictions to a 1
    for row in range(reshaped_target.shape[0]):
        reshaped_target[row,target[row][0]:target[row][1]]=1
    """
    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((target.shape[0], rows - 27))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1
    return reshaped_target

train()
