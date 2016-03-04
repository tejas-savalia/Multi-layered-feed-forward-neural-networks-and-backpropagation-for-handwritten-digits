
# coding: utf-8

# In[1]:

import cPickle, gzip, numpy, math


# Loading the dataset. 50,000 training examples, 10,000 validation examples and 10,000 test examples

# In[2]:

f = gzip.open('E:\Lecs\IIIT\SMAI\Assignments\Assignment 3\mnist.pkl.gz', 'rb')
training_set, validation_set, test_set = cPickle.load(f)
f.close()
print numpy.shape(training_set[0])


# Defining the sigmoid activation function and it's corresponding gradient. Vectorize them to run on the matrix

# In[3]:

def sigmoid(x):
    if x > -6:
        sig = 1/(1+math.exp(-x))
    else:
        sig = 0
    return sig
#sigmoid = lambda x : 1/(1+math.exp(-x))
sigmoid = numpy.vectorize(sigmoid)
def sigmoid_grad(x):
    sig_grad = sigmoid(x)*(1-sigmoid(x))
    return sig_grad
#sigmoid_grad = lambda x : sigmoid(x)*(1-sigmoid(x))
sigmoid_grad = numpy.vectorize(sigmoid_grad)


# Converting the target in vector form. For instance, the number 8 is [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# In[4]:

Y = numpy.zeros((len(training_set[1]), 10))
for (i, j) in zip(Y, training_set[1]) :
    i[j] = 1
#print Y[4]
YTest = numpy.zeros((len(test_set[1]), 10))
for (i, j) in zip(YTest, test_set[1]):
    i[j] = 1
print YTest[1]


# Initializing weights

# In[5]:

epsilon = 0.12
weights1 = numpy.random.uniform(-epsilon, epsilon, size = (8, 784))
weights2 = numpy.random.uniform(-epsilon, epsilon, size = (10, 8))
numpy.shape(weights1)


# Feed-forward Network

# In[6]:

def feed_forward(X, Y, weights1, weights2):
    
    net2 = numpy.dot(X, numpy.transpose(weights1)) + numpy.transpose(numpy.random.uniform(-epsilon, epsilon, size = (8, 1)))
    y2 = sigmoid(net2)
    net3 = numpy.dot(y2, numpy.transpose(weights2)) + numpy.transpose(numpy.random.uniform(-epsilon, epsilon, size = (10, 1)))
    z3 = sigmoid(net3)
    #print z3
    z3Vec = numpy.zeros(10)
    z3Vec[numpy.argmax(z3)] = 1
        
    return (net2, y2, net3, z3Vec)


# Back-Propagation function
# 
# 

# In[7]:


def back_prop(X, Y, num_iters, weights1, weights2):
    #print "Before Back-prop: ", weights2
    w2 = weights2
    D1 = numpy.zeros(numpy.shape(weights1))
    D2 = numpy.zeros(numpy.shape(weights2))
    for i in range(num_iters):
        for j in range(len(X)):
            (net2, y2, net3, z3) = feed_forward(X[j], Y[j], weights1, weights2)
            delta3 = (Y[j] - z3) * sigmoid_grad(net3)
            delWeights2 = numpy.outer(numpy.transpose(delta3), y2)
            delta2 = numpy.dot(delta3, weights2)*sigmoid_grad(net2)
            delWeights1 = numpy.outer(numpy.transpose(delta2), X[j])
            weights1 = weights1 + 0.01*delWeights1
            weights2 = weights2 + 0.01*delWeights2
        print "Iteration", i, "complete"
        #print w2 - weights2
    return (weights1, weights2)


# In[8]:

(w1, w2) = back_prop(training_set[0], Y, 5, weights1, weights2)


# Cross Validation accuracy

# In[9]:

#(a, b, c, d) = feed_forward(test_set[0][4], test_set[1][4], w1, w2)
cross_validation_count = 0.0
for i in range(len(validation_set[0])):
    (a, b, c, d) = feed_forward(validation_set[0][i], validation_set[1][i], w1, w2)
    if numpy.argmax(d) == validation_set[1][i]:
        cross_validation_count = cross_validation_count+1
print "Cross validation Accuracy: ", cross_validation_count/len(validation_set[1])


# Test Accuracy

# In[21]:

test_count = 0.0
confusion_matrix = numpy.zeros((10, 10))
for i in range(len(test_set[0])):
    (a, b, c, d) = feed_forward(test_set[0][i], test_set[1][i], w1, w2)
    if numpy.argmax(d) == test_set[1][i]:
        test_count = test_count + 1
    confusion_matrix[numpy.argmax(d)][test_set[1][i]] += 1 
print "Test set accuracy: ", test_count/len(test_set[1])


# In[23]:

print confusion_matrix

