from keras.models import Sequential
from keras.layers import Convolution2D,  Lambda
from keras.layers import Activation
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as  tf
# with tf.device("/cpu:0"):
'''
A simple script for keras neophyte to study 
the art of neural network surgery

Model accept arbitrary shaped image as input 
and output the probabilistic map of certain pixiel are traffic line 

'''

model = Sequential()
model.add(Convolution2D(6, (5, 5), border_mode='same', activation='relu',name="cnn_11",
                        input_shape=( None, None,3)))
model.add(Convolution2D(1, (3, 3), border_mode='same', activation='relu',name="cnn_112"))
model.add(Convolution2D(1, (3, 3), border_mode='same', activation='relu',name="cnn_113"))
def slice_layer(x):
    return x[..., 0]
model.add(Lambda(slice_layer))
model.add(Activation('softmax'))
model.load_weights("cnn12a.h5",by_name=True)



data2predict = plt.imread('test.jpg').astype('float32')
reshaped_data = data2predict[np.newaxis, ...]
imgpred = model.predict(reshaped_data)[0]
plt.imshow(imgpred)


# Here is functions to analysis keras middle layer
def getIMkeras(model, view_layer):
    input = model.input
    view = model.get_layer(view_layer).output
    func = K.function([input], [view])
    Im = func([reshaped_data])[0]
    # print(np.mean(Im))
    return Im
# Compute intermediate result of first layer
firstLayerOut = getIMkeras(model, 'cnn_11') #cnn_11 is the layer name
# Visualize the 3rd channel of the firstLayerOut
# plt.imshow(firstLayerOut[0,:,:,2])  # UNCOMMENT THIS LINE TO VISUALIZE

# Example for Visualize the weight of the network
weight_bias = model.get_layer('cnn_11').get_weights()
weight = weight_bias[0] # shape should be 5,5,3,6 for conv-layer cnn_11
# UNCOMMENT  LINE BELOW TO VISUALIZE
# plt.imshow(weight[:,:,:,0]) # visualize weight---treating three different channel weight as RGB
