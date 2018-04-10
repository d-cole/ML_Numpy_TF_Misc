from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random as nprandom
import tensorflow as tf
import os
import re
import math
import random
import cPickle
from myalexnet_mod import *

t = int(time.time())
t = 1458432816
#print "t=", t
random.seed(t)


names = ["butler","radcliffe","vartan","bracco","gilpin","harmon"]
one_hot = {'butler':[1, 0, 0, 0 ,0 ,0],'radcliffe':[0, 1, 0, 0 ,0 ,0],'vartan':[0, 0, 1, 0 ,0 ,0],
    'bracco':[0, 0, 0, 1 ,0 ,0],'gilpin':[0, 0, 0, 0 ,1 ,0],'harmon':[0, 0, 0, 0 ,0 ,1]}


def load_images(target_dir,flatten=True):
    """
    Loads the images from target_dir which match the keys in 
        male_keys,female_keys,p6_male_keys and p6_female_keys

    Source for loading images using os.walk()
    http://stackoverflow.com/questions/34426949/using-python-load-images-from-directory-and-reshape
    #From A1
    """
    act = {'butler':[],'radcliffe':[],'vartan':[],'bracco':[],'gilpin':[],'harmon':[]}
    for root, dirnames, filenames in os.walk(target_dir):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|JPG)$",filename):
                image = imread(target_dir + filename);
                image = image[:,:,:].astype(float)/255.0

                #PART 1
                if flatten:
                    image = image.flatten()
                    image = image.reshape(1,image.shape[0])

                #PART 2
                else:
                    image = image.reshape((1,) + image.shape)

                for key in act.keys():
                    if key in filename:
                        act[key].append(image)
    return act


def partition_images(act):
    """
    Partions images in act into training, validation and test sets
    """
    training_set = {'butler':[],'radcliffe':[],'vartan':[],'bracco':[],'gilpin':[],'harmon':[]}
    validation_set = {'butler':[],'radcliffe':[],'vartan':[],'bracco':[],'gilpin':[],'harmon':[]}
    test_set = {'butler':[],'radcliffe':[],'vartan':[],'bracco':[],'gilpin':[],'harmon':[]}

    for key in act.keys():
        im_count = len(act[key]) - 70
        valid_upper_bound = 90
        test_upper_bound = 110

        # 70 images for training set
        training_set[key] = act[key][0:70]
        
        # 20 images for each validation and test set
        validation_set[key] = act[key][70:valid_upper_bound]
        test_set[key] = act[key][valid_upper_bound: test_upper_bound]
 
    #Change so each actors images are matrices 
    for im_set in [training_set,validation_set,test_set]:
        for key in im_set.keys():
            im_set[key] = np.vstack(im_set[key])

    return training_set, validation_set, test_set #test_matrix, test_y_matrix


def dict_to_matrix(act_set):
    """
    Convert a dict of the format {actor:images, actor:images}
        to a matrix of size (num actors)*(num images per actor) X (length of image)
    
    Generates a y_ one-hot encoded matrix that matches 
        the correct values of the actor matrix
    """
    x = []
    y_ = []

    for key in act_set.keys():
        x.append(act_set[key])
        y_ .append([one_hot[key] for i in range(0,len(act_set[key]))])

    return np.vstack(x), np.vstack(y_)


def get_train_batch(n,training_set):
    """
    Randomly picks n images from each actor and generates 
        the y_ correct label set.
    """
    batch_x = []
    batch_y = []

    for actor in training_set.keys():
        rand_idx = random.sample(range(0,70),n)
        for i in rand_idx:
            batch_x.append(training_set[actor][i,:])
            batch_y.append(one_hot[actor])

    batch_x = np.vstack(batch_x)
    batch_y = np.vstack(batch_y)

    return batch_x, batch_y


def make_plot(fig_num,x_vals,y_vals,x_label,y_label,filename):
    """ 
    Plot the given data with axis labels.
    Saves the image as the file specified by filename

    ***From A1 code
    """
    f,axarr = plt.subplots(1,1,sharex=False, sharey=False)
    axarr.plot(x_vals,y_vals)
    axarr.set_xlabel(x_label)
    axarr.set_ylabel(y_label)
    plt.savefig(filename,bbox_inches='tight')


def plot_part1(iterations,train_perf,valid_perf,test_perf):
    """
    Plots learning rates for training, validation, and test sets
    """
    make_plot(0,iterations,train_perf,"Iterations","Correct classification of training set","training_learn_rate.png") 
    make_plot(1,iterations,test_perf,"Iterations","Correct classification of test set","test_learn_rate.png") 
    make_plot(2,iterations,valid_perf,"Iterations","Correct classification of training set","valid_learn_rate.png") 

    return


def part1(in_dimen, num_hidden,training,valid,test):
    """
    One layer fully connected NN.

        input -> hidden tanh layer of num_hidden units -> 
            linear layer -> softmax of linear layer

    This function generates the computation graph, trains it, and returns the session.
    """
    names_matrix = []
    for i in range(0,in_dimen):
        names_matrix.append(names)
    names_matrix = np.vstack(names_matrix) 
    
    num_act = 6

    #Init all variables and run session 
    sess = tf.InteractiveSession()

    #Input placeholder
    # 1 x in_dimen
    x = tf.placeholder(tf.float32, [None, in_dimen])
   
    #Weights from input to hidden layer 
    # in_dimen x num_hidden
    W0 = tf.Variable(tf.random_normal([in_dimen, num_hidden], stddev=0.01))
    # 1 x num_hidden
    b0 = tf.Variable(tf.random_normal([num_hidden], stddev=0.01))
   
    #Weights and bias from hidden layer to output layer 
    # num_hidden x num_act
    W1 = tf.Variable(tf.random_normal([num_hidden, num_act], stddev=0.01))
    # 1 x num_act
    b1 = tf.Variable(tf.random_normal([num_act], stddev=0.01))

    #Layer one is tanh for num_hidden units
    # (1 x in_dimen) * (in_dimen x num_hidden) = (1 x num_hidden)
    layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
    
    #layer2 (output) lin combo of layer1*W1
    # (1 x num_hidden) * (num_hidden x num_act) = (1 x num_act)
    layer2 = tf.matmul(layer1, W1) + b1
   
    #Output layer softmax of layer 2 
    y = tf.nn.softmax(layer2)

    #Define placeholder for one-hot correct answer
    # 1 x num_act
    y_ = tf.placeholder(tf.float32, [None,num_act])

    #Decay coefficient?  - why zero
    lam = 0.01

    #L2 weight penalty
    decay_penalty = lam * tf.reduce_sum(tf.square(W0)) + lam * tf.reduce_sum(tf.square(W1))

    #Reduce cost function + decay_penalty
    NLL = -tf.reduce_sum(y_*tf.log(y)) +  decay_penalty

    # alpha = 0.005, minimize NLL using gradient descent
    alph = 0.001
    train_step = tf.train.GradientDescentOptimizer(alph).minimize(NLL)
   
    init = tf.initialize_all_variables()
    sess.run(init) 

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    iterations = [] 
    test_perf = []
    valid_perf = []
    train_perf = []
    train_set, train_y_ = dict_to_matrix(training)
    valid_set, valid_y_ = dict_to_matrix(valid)
    test_set, test_y_ = dict_to_matrix(test)
    curr_iter = 0

    # TRAINING
    for i in range(3000):
        curr_iter += 1
        batch_xs, batch_ys =  get_train_batch(10,training)

        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        
        if i%5 == 0 :
            test_perf.append(accuracy.eval(feed_dict={x: test_set, y_: test_y_}))
            train_perf.append(accuracy.eval(feed_dict={x: train_set, y_: train_y_}))
            valid_perf.append(accuracy.eval(feed_dict={x: valid_set, y_: valid_y_}))
            iterations.append(curr_iter)

    max_idx = tf.cast(tf.argmax(y,1),tf.int32)
    print "Test performance:" + str(accuracy.eval(feed_dict={x:test_set, y_: test_y_}))
    plot_part1(iterations,train_perf,valid_perf,test_perf)

    return sess, W0, max_idx, x


def generate_conv4_act(act):
    """
    Given each actors images, returns the conv4 outputs from AlexNet for every image.
    """
    conv4_act = {'butler':[],'radcliffe':[],'vartan':[],'bracco':[],'gilpin':[],'harmon':[]}
    sess, conv4, grad, x, name_out = alexNet_sess(act["butler"][0])
    sess.run(tf.initialize_all_variables())

    for actor in act.keys():
        for image in act[actor]:
            image = image.astype(float32)
            conv4_out = sess.run(conv4,feed_dict={x:image})
            print conv4_out.shape
            sys.exit()
            conv4_act[actor].append(conv4_out.flatten())
    
    return conv4_act


def part5(image):
    """
    Gets the gradient of the output y of the modified alexNet with respect 
        to the input image.
    """
    sess,conv4,grad,x,y = alexNet_sess(image)
    sess.run(tf.initialize_all_variables())
    image = image.astype(float32)

    grad_out = np.array(sess.run(grad,feed_dict={x:image}))
    grad_out = grad_out[0,0,:,:,0]
    
    #Remove any negative numbers
    grad_out = grad_out.clip(min=0)

    f,axarr = plt.subplots(1,1,sharex=True,sharey=True)
    axarr.imshow(grad_out)
    plt.savefig("bracco_" + "part5_gradients.png")
    
    return 


def part3(p1_train, p1_valid, p1_test):
    """
    Plot visualizations of two hidden units for networks with 100, 300, and 800 hidden units.
    """
    sess3, W0_300,max_idx, x  = part1(p1_train["butler"].shape[1], 300,p1_train,p1_valid,p1_test)
    sess8, W0_800,max_idx, x  = part1(p1_train["butler"].shape[1], 800,p1_train,p1_valid,p1_test)
    sess1, W0_100,max_idx, x = part1(p1_train["butler"].shape[1], 100,p1_train, p1_valid, p1_test)

    out_W0_300 = sess3.run(W0_300)
    out_W0_800 = sess8.run(W0_800)
    out_W0_100 = sess1.run(W0_100)

    f,axarr = plt.subplots(2,3,sharex=True,sharey=True)
    for i in range(0,2):
        out_W0_100_unit = out_W0_100[:,i].reshape(32,32,3)
        for j in range(0,3):
            axarr[i][j].imshow(out_W0_100_unit[:,:,j], cmap = cm.coolwarm)
    plt.savefig("100_W0plots.png")

    f,axarr = plt.subplots(2,3,sharex=True,sharey=True)
    for i in range(0,2):
        out_W0_300_unit = out_W0_300[:,i].reshape(32,32,3)
        for j in range(0,3):
            axarr[i][j].imshow(out_W0_300_unit[:,:,j], cmap = cm.coolwarm)
    plt.savefig("300_W0plots.png")

    f,axarr = plt.subplots(2,3,sharex=True,sharey=True)
    for i in range(0,2):
        out_W0_800_unit = out_W0_800[:,i].reshape(32,32,3)
        for j in range(0,3):
            axarr[i][j].imshow(out_W0_800_unit[:,:,j], cmap = cm.coolwarm)
    plt.savefig("800_W0plots.png")


if __name__ == "__main__":
    # ---- Part 1 
    act = load_images("./processed/")
    p1_train, p1_valid, p1_test = partition_images(act) 
    #print p1_train["butler"].shape, p1_valid["butler"].shape, p1_test["butler"].shape
    part1(p1_train["butler"].shape[1],300,p1_train,p1_valid,p1_test)
    
    # ---- Part 2
    p2_act = load_images("./large_processed/",flatten=False)
    conv4_act = generate_conv4_act(p2_act)
    conv4_train, conv4_valid, conv4_test = partition_images(conv4_act)
    hidden_units = 300
    sess, W0, names_idx, x = part1(conv4_train["butler"].shape[1], hidden_units, conv4_train, conv4_valid, conv4_test)
    
    # --- P2: Example run - Requires part 2 two run
    #Get conv4 output for an example image of butler
    ex_image_conv4 = conv4_act["butler"][0]
    ex_image_conv4 = ex_image_conv4.reshape(1,ex_image_conv4.shape[0])
    #Run image through trained session
    print names[sess.run(names_idx, feed_dict={x:ex_image_conv4})]

    # ---- Part 3 - requires part 1 to run
    part3(p1_train,p1_valid,p1_test)

    # --- Part 5 Example - requires part 2 to run
    ex_image = p2_act["butler"][0]     
    sess,conv4,grad,x,names_idx = alexNet_sess(ex_image)
    print names[sess.run(names_idx, feed_dict={x:ex_image})]

    # --- Part 5 - requires part 2 to run
    part5(p2_act["bracco"][10])
    
    
