import numpy as np
import matplotlib.pyplot as plt

from init_network_parameters import *
from init_fpga_parameters import *
from activation_functions import sigmoid, sigmoid_backward, relu, relu_backward

# Seed Fixed
np.random.seed(10)

def initialize_parameters(layer_type):
    parameters, parameters_float = {}, {}

    assert ((layer_type=='input') or (layer_type=='output'))

    if layer_type == 'input':

        parameters['W'] = np.random.randn(HIDDEN_LAYERS_SIZE[0], INPUT_LAYER_SIZE) * 0.01
        parameters_float['W'] = parameters['W'] * pow(2, COEFF_INPUT_W_SIZE)
        parameters['W'] = parameters_float['W'].astype('int16')

        parameters['b'] = np.zeros((HIDDEN_LAYERS_SIZE[0], 1), dtype='int16')

    else:
        parameters['W'] = np.random.randn(OUTPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZE[-1]) * 0.01
        parameters_float['W'] = parameters['W'] * pow(2, COEFF_OUTPUT_W_SIZE)
        parameters['W'] = parameters_float['W'].astype('int16')

        parameters['b'] = np.zeros((OUTPUT_LAYER_SIZE, 1), dtype='int16')

    return parameters

# parameters_input = initialize_parameters(layer_type='input')
# parameters_output = initialize_parameters(layer_type='output')

# print("W_input = " + str(parameters_input["W"]))
# print("b_input = " + str(parameters_input["b"]))
# print("W_output = " + str(parameters_output["W"]))
# print("b_output = " + str(parameters_output["b"]))




def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters, parameters_float = {}, {}

    L = HIDDEN_LAYERS_COUNT            # number of layers in the network

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters_float['W' + str(l)] = parameters['W' + str(l)] * pow(2, COEFF_HIDDEN_W_SIZE[l])
        parameters['W' + str(l)] = parameters_float['W' + str(l)].astype('int16')

        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1), dtype='int16')
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# parameters_hidden = initialize_parameters_deep(HIDDEN_LAYERS_SIZE)

# print("W1 = " + str(parameters_hidden["W1"]))
# print("b1 = " + str(parameters_hidden["b1"]))
# print("W2 = " + str(parameters_hidden["W2"]))
# print("b2 = " + str(parameters_hidden["b2"]))



# # ## 4 - Forward propagation module
# # 
# # ### 4.1 - Linear Forward 


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# A = np.ones((INPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZE[0]), dtype=int)
# W, b = parameters_input["W"], parameters_input["b"]
# print("A = " + str(A))
# print("W = " + str(W))
# print("b = " + str(b))

# Z, linear_cache = linear_forward(A, W, b)

# print("Z = " + str(Z))


# # ### 4.2 - Linear-Activation Forward
# # 
def linear_activation_forward(A_prev, W, b, activation, output_size):
#     """
#     Implement the forward propagation for the LINEAR->ACTIVATION layer

#     Arguments:
#     A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
#     W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
#     b -- bias vector, numpy array of shape (size of the current layer, 1)
#     activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
#     output_size - bit width of output data

#     Returns:
#     A -- the output of the activation function, also called the post-activation value 
#     cache -- a python tuple containing "linear_cache" and "activation_cache";
#              stored for computing the backward pass efficiently
#     """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z, output_size)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z, output_size)

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

A_prev = np.ones((INPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZE[0]), dtype=int)
W, b = parameters_input["W"], parameters_input["b"]
print("A = " + str(A))
print("W = " + str(W))
print("b = " + str(b))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, 
                                                        activation = "sigmoid",
                                                        output_size = OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE[0])
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, 
                                                        activation = "relu",
                                                        output_size = OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE[0])
print("With ReLU: A = " + str(A))


# # ### L-Layer Model 

def L_model_forward(X, parameters_input, parameters_hidden, parameters_output):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters_input -- W and b for input layer
    parameters_hidden -- W and b for hidden layer
    parameters_output -- W and b for output layer
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A_prev = X

    # Input layer
    A, cache = linear_activation_forward(A_prev, parameters_input["W"], parameters_input["b"], 
                                                activation = INPUT_ACTIVATION,
                                                output_size = OUTPUT_AFTER_INPUT_LAYER_DATA_SIZE)

    caches.append(cache)

    # Hidden layers
    L = HIDDEN_LAYERS_COUNT                 # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 

        A, cache = linear_activation_forward(A_prev, parameters_hidden["W"+str(l)], parameters_hidden["b"+str(l)], 
                                                activation = HIDDER_LAYERS_ACTIVATIONS[l],
                                                output_size = OUTPUT_AFTER_HIDDEN_LAYER_DATA_SIZE[l])

        caches.append(cache)

    
    # Output layer
    AL, cache = linear_activation_forward(A, parameters_output["W"], parameters_output["b"],
                                                activation = OUTPUT_ACTIVATION,
                                                output_size = OUTPUT_LAYER_DATA_SIZE)
    
    caches.append(cache)

    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# X = np.ones((INPUT_LAYER_SIZE, HIDDEN_LAYERS_SIZE[0]), dtype=int)

# AL, caches = L_model_forward(X, parameters_input, parameters_hidden, parameters_output)

# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))

# # ## 5 - Cost function

def compute_cost(AL, Y, log_size, cost_size):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    log_size -- bit width of data after np.log
    cost_size -- bit width of output data

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    log_AL = np.log(AL)
    log_AL_float = log_AL * pow(2, log_size)
    log_AL = log_AL_float.astype('int16')

    log_min_AL = np.log(1-AL)
    log_min_AL_float = log_min_AL * pow(2, log_size)
    log_min_AL = log_min_AL_float.astype('int16')

    # Compute loss from aL and y.

    cost = (-1/m) * np.sum(np.multiply(Y,log_AL) + np.multiply((1-Y),log_min_AL))

    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

# AL, _ = L_model_forward(X, parameters_input, parameters_hidden, parameters_output)
# AL = 0.1
# Y  = np.ones((1,1), dtype='int16')

# print("cost = " + str(compute_cost(AL, Y, log_size=LOG_COST_DATA_SIZE, cost_size=COST_DATA_SIZE)))