from pylab import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import sys
import cPickle
import os
from scipy.io import loadmat

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

one_hot = [np.arange(10)*0 for i in range(0,10)]
for i in range(0,10):
    one_hot[i][i] = 1


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))


def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W, y.T)+b)


def forward(x, W0, b0, W1, b1):
    """
    Run x through the neural network.
    """
    L0 = tanh_layer(x, W0, b0)
    L1 = np.dot(W1, L0)

    output = softmax(L1)
    return L0, L1, output

    
def cost(y, y_):
    return -sum(y_*log(y)) 


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''

    B1_sum = np.ones((x.shape[0],1))        
    dCdL1 =  (y - y_) #oi layer
    dCdW1 =  dot(L0, dCdL1.T) #if you don't want the nonlinearity at the top layer
    dCdB1 =  dot(dCdL1,B1_sum) #Sum across cases of B

    B0_sum = np.ones((x.shape[0],1))
    dCdL0 = (np.dot(dCdL1.T, (W1)))
    dCdW0 = np.dot(dCdL0.T, x)
    dCdB0 = np.dot(dCdL0.T,B0_sum)

    return dCdW1.T, dCdB1, dCdW0, dCdB0


def part8(W0,b0,W1,b1,x,y_,dCdW0,dCdB0):
    """
    Displays both the precise and approximate gradient 
    """
    h = 0.000001
    for i in range(653,656):
        cx = cost(forward(x,W0,b0,W1,b1)[2],y_)
        W0[0,i] += h
        cxh = cost(forward(x,W0,b0,W1,b1)[2],y_)
        print "Finite W0[0:" + str(i) + "]: " + str((cxh-cx)/h)
        print "Precise dCdW0[0:" + str(i) + "]: " + str(dCdW0[0,i])

    for i in range(0,2):
        cx = cost(forward(x,W0,b0,W1,b1)[2],y_)
        b0[i] += h
        cxh = cost(forward(x,W0,b0,W1,b1)[2],y_)
        print "Finite b0[" + str(i) + "]: " + str((cxh-cx)/h)
        print "Precise dCdB0[" + str(i) + "]: " + str(dCdB0[i])


def part1():
    """
    Plot 10 images of each digit
    http://matplotlib.org/examples/pylab_examples/subplots_demo.html
    """
    num_rows = 10
    num_cols = 10

    f,axarr = plt.subplots(num_rows,num_cols, sharex=True, sharey=True)
    for i in range(0,num_rows):
        for j in range(0,num_cols):
            axarr[i,j].imshow(M["train"+str(i)][str(j)].reshape((28,28)))
            axarr[i,j].axis('off')

    plt.savefig("part1_fig.png",bbox_inches='tight')


def make_plot(fig_num,x_vals,y_vals,x_label,y_label,filename,in_label,lim_y):
    """ 
    Plot the given data with axis labels. Legend/data labels optional.
    Saves the image as the file specified by filename

    *From A1 code
    """

    fig = figure(fig_num)  
    if lim_y:
        ylim([0,1])

    xlabel(x_label)
    ylabel(y_label)

    if in_label != None:
        plot(x_vals, y_vals,label=in_label)
        legend(loc='upper right',shadow=True)
    else:
        plot(x_vals, y_vals)
        
    print "Figure " + filename + " saved."
    savefig(filename,bbox_inches='tight')


def plot_subplot(rows,columns,data,filename):
    """
    Plot a subplot of rowXcolumn images
    """
    f,axarr = plt.subplots(rows,columns, sharex=True, sharey=True)
    for j in range(0,columns):
        axarr[j].imshow(data[:,j].reshape((28,28)))
        axarr[j].axis('off')

    plt.savefig(filename,bbox_inches='tight')


def plot_part5_9(y,y_,X,iterations,test_performance,test_prob,train_performance,train_prob,fig_num,file_prefix):
    """
    Plot 20 correctly classified digits and 10 incorrectly classified digits
    &
    Plot performance and cost vs. iteration count
    """
    #Get lists of correctly and incorrectly classified digits
    correct_digits = []
    incorr_digits = []
    i = 0
    while(i < X.shape[0]):
        if(y[:,i].argmax()) == (y_[:,i].argmax()):
            correct_digits.append(X[i,:])
        else:
            incorr_digits.append(X[i,:])     
        i+=1
    
    #Shuffle list so first 20 numbers are not all 0s 
    shuffle(correct_digits)    
    shuffle(incorr_digits)

    #Plot images
    plot_subplot(1,20,np.vstack(correct_digits[0:20]).T,file_prefix + "_correct.png")
    plot_subplot(1,10,np.vstack(incorr_digits[0:10]).T,file_prefix + "_incorr.png")
     
    #Plot performance & cost
    make_plot(fig_num,iterations,test_performance,"Iterations","Performance on test set",file_prefix + "_test_perf.png",None,True) 
    make_plot(fig_num+10,iterations,test_prob,"Iterations","Cost",file_prefix + "_test_perfCost.png",None,False)
    make_plot(fig_num+3,iterations,train_performance,"Iterations","Performance on test set",file_prefix + "_train_perf.png",None,True) 
    make_plot(fig_num+50,iterations,train_prob,"Iterations","Cost",file_prefix + "_train_perfCost.png",None,False)


def part2(x,W,b):
    """
    Computes o_i = SUM_j w_ji + b_i. 
    """
    y = (np.dot(W,np.transpose(x)) + b)
    s_max = softmax(y)
    return s_max


def dCdW(x,y,y_):
    """
    Computes the gradient of the cost function with respect to the weights.
    """
    return np.dot(y - y_,x)


def dCdB(x,y,y_):
    """
    Computes the gradient of the cost function with respect to the biases
    """
    sum_B_cases = np.ones((x.shape[0],1))        
    return np.dot(((y) - y_), sum_B_cases)


def deriv_finit_approx_W(W,X,i,j,y_,b):
    """
    Compute the finite approximation of the deriviative 
        of the cost with respect to weight i,jk.
    """
    cx = cost(part2(X,W,b),y_) 
    W[i,j] = W[i,j] +  0.00001
    cxh = cost(part2(X,W,b),y_)
    
    return (cxh - cx)/0.00001


def deriv_finit_approx_B(W,X,i,y_,b):
    """
    Compute the finite approximation of the derivative of the cost
        function with respect to bi
    """
    cx = cost(part2(X,W,b),y_) 
    b[i] = b[i] + 0.000001
    cxh = cost(part2(X,W,b),y_)
    
    return (cxh - cx)/0.000001

def get_all_set(set_type):
    """
    Returns...
    """
    set_array = []
    for i in range(0,10):
        for j in range(0,len(M[set_type + str(i)])):
            set_array.append(M[set_type + str(i)][j].reshape(28,28).flatten()/255.0)

    return np.vstack(set_array)


def gen_batch_set(set_type,num_per_number,start_idx):
    """
    Generates training set X grabbing num_per_number arrays per number starting at start_idx
    """
    X_arr = []
    for num in range(0,10):
        for i in range(0,num_per_number):
            X_arr.append(M[set_type + str(num)][(start_idx + i)].reshape(28,28).flatten()/255.0)
 
    X = np.vstack(X_arr) 
    return X

    
def gen_label_set(num_per_number):
    """
    Generates matrix y_ of one_hot encodings
    """
    y_ = np.empty((10,num_per_number*10))
    for num in range(0,10):
        for i in range(0,num_per_number):
            y_[:,num_per_number*num + i] = np.transpose(one_hot[num])

    return y_


def get_performance(y,y_):
    """
    Return the correct rate of classification for the output y,
        and the target output y_.
    """
    correct = 0.0
    total = y.shape[1]
    for i in range(y.shape[1]):
        if (y[:,i].argmax()) == (y_[:,i].argmax()):
            correct += 1.0

    return correct/total


def part4(W,X,y_,b,dW,dB):
    """
    Compare the estimated gradient to precise gradient.
    """
    for j in range(600,X.shape[1]):
        approx = deriv_finit_approx_W(W,X,0,j,y_,b)
        print "Weight: W[0:" + str(j) + "]"
        print "Weight - Finite estimate: " + str(approx)  
        print "Weight - Precise derivative: " + str(dW[0,j]) 
       
    for j in range(0,2): 
        print "Bias: B[" + str(j) + "]"
        approx = deriv_finit_approx_B(W,X,j,y_,b)
        print "Bias - Finite estimate: " + str(approx)
        print "Bias - Precise derivative: " + str(dB[j]) 


def plot_part6(W):
    """
    Plots the heat maps of the W's
    """
    num_rows = 1
    num_cols = 10
    f,axarr = plt.subplots(num_rows,num_cols, sharex=True, sharey=True)

    for j in range(0,num_cols):
        axarr[j].imshow(W[j,:].reshape(28,28))
        axarr[j].axis('off')
    
    plt.savefig("part6_fig.png",bbox_inches='tight')


def part5(training_set,test_set):
    """
    Minimizes cost function use mini-batch gradient descent.
    """
    batchsize = 5 #Per number - 50 total

    #Add array to track iterations and performance on train & test set
    iterations = []
    test_performance = []
    test_costs = []
    train_performance = []
    train_costs = []

    iteration = 0 
    epoch = 0
    total_iterations = 0
    l_rate = 0.01
    batch_start_pos = 0    

    y_ = gen_label_set(batchsize)
    train_y_ = gen_label_set(training_set.shape[0]/10)
    test_y_ = gen_label_set(test_set.shape[0]/10)

    #initialize small random weights
    W = np.random.rand(10,784)*0.0001 
    b = np.random.rand(10,1)*0.0
    test_perf = 0     
    #Removed performance < 0.95
    #TODO fix iteration max
    while(test_perf < 0.89):
        iteration = 0
        batch_start_pos = 0
        while(iteration*batchsize < 1084):
            #Get first batch of 50 images per number
            X = gen_batch_set("train",batchsize,batch_start_pos)
            batch_start_pos += batchsize
    
            #Run batch through network
            y = part2(X,W,b)
          
            #Get weight & bias gradient 
            dW = dCdW(X,y,y_)
            dB = dCdB(X,y,y_)
    
            ##Adjust weights
            W = W - l_rate*dW
    
            ##Adjust biases
            b = b - l_rate*dB

            if total_iterations%50 == 0:
                ##Evaluate performance 
                train_y = part2(training_set,W,b)
                train_perf = get_performance(train_y,train_y_)
                train_cost = cost(train_y,train_y_)        
    
                #Get performance on test set
                test_y = part2(test_set,W,b)
                test_perf = (get_performance(test_y,test_y_))
                test_cost = cost(test_y,test_y_)
                #Update arrays for plotting
                iterations.append(total_iterations)
                test_performance.append(test_perf)
                train_performance.append(train_perf) 
    
                test_costs.append(test_cost)
                train_costs.append(train_cost)

            iteration += 1        
            total_iterations += 1 

        epoch += 1
   
    #Plot performance and correct/incorrectly classified characters 
   
    part4(W + l_rate*dW,X,y_,b + l_rate*dB,dW,dB) 
    plot_part5_9(test_y,test_y_,test_set,iterations,test_performance,test_costs,train_performance,train_costs,16,"p5_")
    plot_part6(W)


def part9(training_set,test_set):
    """
    Performs mini-batch gradient descent on network for part 9.
    """
    #Per number
    batchsize = 5    

    iterations = []
    test_performance = []
    test_costs = []
    train_performance = []
    train_costs = []

    iteration = 0
    total_iterations = 0
    batch_start_pos = 0
    l_rate = 0.01

    #initialize small random weights
    W0 = np.random.rand(300,784)*0.001 
    W1 = np.random.rand(10,300)*0.001
    b0 = np.random.rand(300,1)*0.001
    b1 = np.random.rand(10,1)*0.001
    y_ = gen_label_set(5)

    test_y_ = gen_label_set(test_set.shape[0]/10)
    train_y_ = gen_label_set(training_set.shape[0]/10)
    
    test_perf = 0
    epoch = 0
    while(test_perf < 0.98):
        iteration = 0
        batch_start_pos = 0
        while (iteration*batchsize < 5400):
            X = gen_batch_set("train",batchsize,batch_start_pos)
            batch_start_pos += batchsize

            #Run network
            L0,L1,y = forward(X,W0,b0,W1,b1)

            #Get gradient for weights and biases        
            dCdW1, dCdB1, dCdW0, dCdB0 = deriv_multilayer(W0,b0,W1,b1,X,L0,L1,y,y_)

            #Adjust weights and biases
            W0 = W0 - l_rate*(dCdW0)
            W1 = W1 - l_rate*(dCdW1)
            b0 = b0 - l_rate*(dCdB0)
            b1 = b1 - l_rate*(dCdB1)

            if(total_iterations%50 == 0):
                #Get performance on training set
                train_y = forward(training_set,W0,b0,W1,b1)[2]
                train_perf = get_performance(train_y,train_y_)
                train_cost = cost(train_y,train_y_)

                #Get performance on test set
                test_y = forward(test_set,W0,b0,W1,b1)[2]
                test_perf = (get_performance(test_y,test_y_))
                test_cost = cost(test_y,test_y_)

                #Update arrays for plotting
                iterations.append(total_iterations)
                test_performance.append(test_perf)
                train_performance.append(train_perf) 
                test_costs.append(test_cost)
                train_costs.append(train_cost)
            
             
            total_iterations +=1
            iteration += 1
        epoch += 1
    
    #Evaluate test set to plot correct vs. incorrect characters
    part8(W0 + l_rate*(dCdW0),b0 + l_rate*(dCdB0),W1 + l_rate*(dCdW1),b1 + l_rate*(dCdB1),X,y_,dCdW0,dCdB0)
    plot_part5_9(test_y,test_y_,test_set,iterations,test_performance,test_costs,train_performance,train_costs,10,"p9_")
    part10(W0)


def part10(W0):
    """
    Plot weights to all Hjs
    """ 
    #plots all hjs
    f,axarr = plt.subplots(10,3,sharex = True,sharey = True)
    for i in range(0,10):
        for j in range(0,3):
            axarr[i,j].imshow(W0[i*3 + j,:].reshape((28,28)))
            axarr[i,j].axis('off') 

    plt.savefig("part10_fig.png",bbox_inches='tight')


if __name__ == "__main__":
    part1()

    #Get all cases for training and test set
    train_set = get_all_set("train")
    test_set = get_all_set("test")

    part5(train_set,test_set) 
    part9(train_set,test_set) 
    


   


