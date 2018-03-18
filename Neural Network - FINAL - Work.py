import numpy as np



alphas = [10]

# compute sigmoid nonlinearity
def sigmoid(x):
        output = 1 / (1 + np.exp(-x))
        return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

X = np.array([[1, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[1],
              [0],
              [1],
              [1]])

for alpha in alphas:
    print ("\n\nARTIFICIAL NEURAL NETWORK\nBinary Input and Output ANN with One Hidden Layer\n")
    print ("\nTraining With Alpha:" + str(alpha))
    np.random.seed(1)
    # randomly initialize our weights with mean 0
    synapse_0 = np.random.random((3, 4))
    synapse_1 = np.random.random((4, 1))

    prev_synapse_0_weight_update=np.zeros_like(synapse_0)
    prev_synapse_1_weight_update=np.zeros_like(synapse_1)
    synapse_0_direction_count=np.zeros_like(synapse_0)
    synapse_1_direction_count=np.zeros_like(synapse_1)

for j in range (60000):
    #Feed forward through layers 0, 1, and 2
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    
    t_X = layer_0
    t_layer_1 = layer_1
    t_layer_2 = layer_2
    
# how much did we miss the target value?

    layer_2_error =  y - layer_2
    
    t_layer_2_error =  layer_2_error

    t_layer_2_error_mean = np.mean(np.abs(layer_2_error))

    if (j % 10000) == 0:
        print ("Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
        # in what direction is the target value?

        # were we really sure? if so, don't change too much.
        
    layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?

    layer_1_error = layer_2_delta.dot(synapse_1.T)


    # in what direction is the target l1?

    # were we really sure? if so, don't change too much.
    
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
    synapse_0_weight_update =  (layer_0.T.dot(layer_1_delta))

    if (j > 0):

     synapse_1 += alpha * synapse_1_weight_update
     synapse_0 += alpha * synapse_0_weight_update


print ("Synapse 0")
print (synapse_0)
print ("Synapse 1")
print (synapse_1)


#Feed forward through layers 0, 1, and 2
layer_0 = [0, 0, 1]
layer_1 = sigmoid(np.dot(layer_0, synapse_0))
layer_2 = sigmoid(np.dot(layer_1, synapse_1))
print(layer_2)
print("\nFor Input [0, 0, 1] the Output is")
if layer_2 > 0.75:
        print ('1')
else:
        print ('0')
print("\n")
print("t_X")
print(t_X)
print("\n")
print("t_layer_1")
print(t_layer_1)
print("\n")
print("t_layer_2")
print(t_layer_2)
print("\n")
print("t_layer_2_error")
print(t_layer_2_error)
print("\n")
print("t_layer_2_error_mean")
print(t_layer_2_error_mean)
layer_2_error
print("\n")
print("layer_2_error")
print(layer_2_error)
print("\n")
print("mean_layer_2_error")
print(np.mean(layer_2_error))
               
print("\n")
print("synapse_1_weight_update")
print(synapse_1_weight_update)
print("\n")
print("synapse_0_weight_update")
print(synapse_0_weight_update)

print("\n")
print("layer_0")
print(layer_0)
print("\n")
print("synapse_0")
print(synapse_0)
print("\n")
print("layer_1")
print(layer_1)
print("\n")
print("synapse_1")
print(synapse_1)
print("\n")
print("layer_2")
print(layer_2)
print("\n")
test_1 = sigmoid(np.dot(layer_0, synapse_0))
print("test_1")
print(test_1)
print("\n")
print("test_2")
test_2 = sigmoid(np.dot(X, synapse_0))
print(test_2)
input ('')

