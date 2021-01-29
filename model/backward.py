import numpy as np
import matplotlib.pyplot as plt

from init_network_parameters import *
from init_fpga_parameters import *
from activation_functions import sigmoid, sigmoid_backward, relu, relu_backward

# Seed Fixed
np.random.seed(10)

# # ## 6 - Backward propagation module


# # ### 6.1 - Linear backward
# # 

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ,A_prev.T)/m).astype('int16')
    db = (np.sum(dZ,axis=1,keepdims=True)/m).astype('int16')
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

# dZ = np.ones((3,3), dtype='int16')
# linear_cache = np.ones((3,3), dtype='int16') * 2, np.ones((3,3), dtype='int16') * 2, np.ones((3,1), dtype='int16')

# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
print ("db = " + str(db))

# # ### 6.2 - Linear-Activation backward

def linear_activation_backward(dA, cache, activation, output_size):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    output_size -- bit width of output data
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache = cache [0:-1]
    activation_cache = cache[-1]

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache, output_size)
        print(dZ.shape)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# dAL = np.ones((3,3), dtype='int16')

# linear_cache = [np.ones((3,3), dtype='int16') * 2, np.ones((3,3), dtype='int16') * 3, np.ones((3,1), dtype='int16')*4]
# activation_cache = [np.ones((3,3), dtype='int16') * 5]

# linear_activation_cache = linear_cache + activation_cache

# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, \
#                                             activation = "sigmoid", \
#                                             output_size = OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE)
# print ("sigmoid:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db) + "\n")

# dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, \
#                                             activation = "relu", \
#                                             output_size = OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE)
# print ("relu:")
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))

# # ### 6.3 - L-Model Backward 


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = HIDDEN_LAYERS_COUNT # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Output layer gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[0:4]
    grads["dA" + str(L)], grads["dW" + str(L+1)], grads["db" + str(L+1)] = \
            linear_activation_backward(dAL, current_cache, 'sigmoid', OUTPUT_AFTER_OUTPUT_BACKLAYER_DATA_SIZE)


    # Loop from hidden layers
    for l in reversed(range(1, L)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[4*l:4*(l+1)]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu', OUTPUT_AFTER_HIDDEN_BACKLAYER_DATA_SIZE[l-1])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp


    # Input layer
    current_cache = caches[4*(L+1+1):]
    grads["dA" + str(0)], grads["dW" + str(1)], grads["db" + str(1)] = \
            linear_activation_backward(dA_prev_temp, current_cache, 'relu', OUTPUT_AFTER_INPUT_BACKLAYER_DATA_SIZE)

    return grads

# AL = np.ones((1,3), dtype='int16') * 3
# Y_assess = np.ones((1,3), dtype='int16') * 3
# caches = linear_activation_cache * (HIDDEN_LAYERS_COUNT + 3)

# grads = L_model_backward(AL, Y_assess, caches)
# print(grads)

# # ### 6.4 - Update Parameters
# # 
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = HIDDEN_LAYERS_COUNT # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

# parameters = {}

# parameters["W1"] = np.ones((3,3), dtype='int16')
# parameters["b1"] = np.ones((1,3), dtype='int16')
# parameters["W2"] = np.zeros((3,3), dtype='int16')
# parameters["b2"] = np.zeros((1,3), dtype='int16')
# parameters["W3"] = np.ones((3,3), dtype='int16')
# parameters["b3"] = np.ones((1,3), dtype='int16')

# parameters = update_parameters(parameters, grads, 0.1)

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))
# print ("W3 = "+ str(parameters["W3"]))
# print ("b3 = "+ str(parameters["b3"]))