import cv2
import pdb
import os.path
import numpy as np
from os import path
from keras.models import Sequential, load_model
from keras.layers import Dense


def init_data():
    train_data = np.load('train_imgs.npy')
    labels_data = np.load('labels.npy',)
    
    
    predictions = np.array([cv2.imread('data\\51.jpg', cv2.IMREAD_GRAYSCALE),
                            cv2.imread('data\\52.jpg', cv2.IMREAD_GRAYSCALE),
                            cv2.imread('data\\53.jpg', cv2.IMREAD_GRAYSCALE),
                            cv2.imread('data\\54.jpg', cv2.IMREAD_GRAYSCALE),
                            cv2.imread('data\\55.jpg', cv2.IMREAD_GRAYSCALE)])
    
    #y_labels = [436, 428, 424, 417, 417, 415, 411, 409, 406, 404, 402, 407, 410, 410, 410, 409, 409, 408, 398, 388, 377, 366, 357, 349, 340, 334, 325, 318, 314, 309, 303, 298, 294, 289, 286, 282, 279, 275, 272, 269, 267, 265, 262, 261, 260, 257, 255, 255, 253, 257]
    #x_labels = [y - 35 for y in y_labels]
    #pdb.set_trace()
    #new_labels = np.array([x_labels, y_labels]).T
    #np.save('labels.npy', new_labels)
    labels_data = labels_data[:,0]
    rows = train_data.shape[1]
    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
    predictions = predictions.reshape((predictions.shape[0], predictions.shape[1] * predictions.shape[2]))
    
    return train_data[:50], labels_data, predictions, rows
    
def train():
    X, y, predictions, rows = init_data()
    y = reshape_data(y, rows)

    model = Sequential()
    if not path.exists('fish_id.h5'):
        model.add(Dense(100, input_shape=(22000,), activation='tanh'))
        model.add(Dense(550, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, y, epochs=250, batch_size=10, verbose=0)
        #model.save('fish_id.h5')
    else:
        model = load_model('fish_id.h5')

    #results = model.evaluate(Xtest, ytest, batch_size=5)
    #print('test loss, test acc:', results)
    
    preds = model.predict(predictions)
    print(np.where(preds==1))
    pdb.set_trace()
    
def reshape_data(target, rows):
    #Reshape from from just a # to all 0's
    reshaped_target = np.zeros((target.shape[0], rows))

    #Change index of correct predictions to a 1
    reshaped_target[np.arange(reshaped_target.shape[0]), target]=1

    return reshaped_target

train()
