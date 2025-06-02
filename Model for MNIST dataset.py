import sys
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats, optimize
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time

def sigmoid(u):
    ''' In this function, we will return sigmoid of u'''
    # compute sigmoid(z) and return
    return 1.0/(1.0+np.exp(-u))

def initializer(dim):
    """
    Initialise the weights and the bias to tensors of dimensions (dim,1) for W and
    to 1 for B (a scalar)
    """
    W = np.zeros((dim,1))
    B = 0
    return W,B

def ForwardBackProp(X, Y, W, B):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    W -- weights, a numpy array of size (num_px * num_px, 1) 
    B -- bias, a scalar
    X -- data of size (num_px * num_px, number of examples)
    Y -- true "label" vector of size (1, number of examples)

    Return:
    J -- negative log-likelihood cost for logistic regression
    dW -- gradient of the loss with respect to w, thus same shape as w
    dB -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[0]
    dW = np.zeros((W.shape[0],1))
    dB = 0

    Z = np.dot(X,W)+B
    Yhat = sigmoid(Z) 
    J = -(1/m)*(np.dot(Y.T,np.log(Yhat))+np.dot((1-Y).T,np.log(1-Yhat)))
    dW = (1/m)*np.dot(X.T,(Yhat-Y))
    dB = (1/m)*np.sum(Yhat-Y)
    return J, dW, dB

def predict(X,W,B):
    '''
    Predict whether the label is 0 or 1 
    '''
    Z = np.dot(X,W)+B
    Yhat_prob = sigmoid(Z)
    Yhat = np.round(Yhat_prob).astype(int)
    return Yhat, Yhat_prob

def gradient_descent(X, Y, W, B, alpha, max_iter):
    """
    This function optimizes w and b by running a gradient descent algorithm
    """
    i=0
    RMSE = 1
    cost_history=[]
    
    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    while (i<max_iter)&(RMSE>10e-6):
        J, dW, dB = ForwardBackProp(X,Y,W,B)
        W = W - alpha*dW
        B = B - alpha*dB
        cost_history.append(J)
        Yhat, _ = predict(X,W,B)
        RMSE = np.sqrt(np.mean(Yhat-Y)**2)
        i+=1
        if i%50==0:
            sys.stdout.write("=")
            sys.stdout.flush()
    sys.stdout.write("]\n") # this ends the progress bar
    return cost_history, W, B, i

def stochastic_gradient_descent(X, Y, W, B, alpha, max_iter):
    i=0
    RMSE = 1
    cost_history=[]
    m = X.shape[0]
    batchsize = 10
    
    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    while (i<max_iter)&(RMSE>10e-6):
        inds = np.random.choice(m, batchsize)
        J, dW, dB = ForwardBackProp(X[inds],Y[inds],W,B)
        W = W - alpha*dW
        B = B - alpha*dB
        cost_history.append(J)
        Yhat, _ = predict(X,W,B)
        RMSE = np.sqrt(np.mean(Yhat-Y)**2)
        i+=1
        if i%50==0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    sys.stdout.write("]\n") # this ends the progress bar
    return cost_history, W, B, i

def implicit_stochastic_gradient_descent(X, Y, W, B, alpha, max_iter):
    i=0
    RMSE = 1
    cost_history=[]
    m = X.shape[0]
    batchsize = 10
    
    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    while (i<max_iter)&(RMSE>10e-6):
        inds = np.random.choice(m, batchsize)
        B_old = B
        J, dW, dB = ForwardBackProp(X[inds],Y[inds],W,B)
        W = W - alpha*dW
        B_sgd = B - alpha*dB
        f_b = lambda B: B - (B_old - alpha * ForwardBackProp(X[inds],Y[inds],W,B)[2])
        B = optimize.root_scalar(f_b, x0 = B_sgd).root
        Yhat, _ = predict(X,W,B)
        RMSE = np.sqrt(np.mean(Yhat-Y)**2)
        i+=1
        if i%50==0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    sys.stdout.write("]\n") # this ends the progress bar
    return cost_history, W, B, i

def adam(X, Y, W, B, alpha, max_iter):
    i=0
    RMSE = 1
    cost_history=[]
    m = X.shape[0]
    batchsize = 10
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    first_moment_W = 0
    second_moment_W = 0
    first_moment_B = 0
    second_moment_B = 0
    
    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
    
    while (i<max_iter)&(RMSE>10e-6):
        inds = np.random.choice(m, batchsize)
        J, dW, dB = ForwardBackProp(X[inds],Y[inds],W,B)
        alpha_t = alpha * np.sqrt(1-beta_2**i)/(1-beta_1**i + epsilon)
        first_moment_W = beta_1 *first_moment_W + (1-beta_1)*dW
        second_moment_W = beta_2 *second_moment_W + (1-beta_2)*(dW**2)
        W = W - alpha_t*first_moment_W/(np.sqrt(second_moment_W) + epsilon)

        first_moment_B = beta_1 *first_moment_B + (1-beta_1)*dB
        second_moment_B = beta_2 *second_moment_B + (1-beta_2)*(dB**2)
        B = B - alpha_t*first_moment_B/(np.sqrt(second_moment_B) + epsilon)

        cost_history.append(J)
        Yhat, _ = predict(X,W,B)
        RMSE = np.sqrt(np.mean(Yhat-Y)**2)
        i+=1
        if i%50==0:
            sys.stdout.write("=")
            sys.stdout.flush()
    
    sys.stdout.write("]\n") # this ends the progress bar
    return cost_history, W, B, i

def LogRegModel(X_train, X_test, Y_train, Y_test, alpha, max_iter, method):
    nbr_features = X_train.shape[1]
    W, B = initializer(nbr_features)
    if method == "SGD":    
        cost_history, W, B, i = stochastic_gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    elif method == "ISGD":
        cost_history, W, B, i = implicit_stochastic_gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    elif method == "ADAM":
        cost_history, W, B, i = adam(X_train, Y_train, W, B, alpha, max_iter)
    else:
        cost_history, W, B, i = gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    Yhat_train, _ = predict(X_train, W, B)
    Yhat, _ = predict(X_test, W, B)
    
    train_accuracy = accuracy_score(Y_train, Yhat_train)
    test_accuracy = accuracy_score(Y_test, Yhat)
    conf_matrix = confusion_matrix(Y_test, Yhat, normalize='true')
    
    model = {"Weights": W,
             "Bias": B,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_prediction": Yhat,
            "confusion_matrix": conf_matrix,
            "cost_history": cost_history}
    return model

def bias_variance_decomp(X_train, y_train, X_test,y_test, alpha, max_iter, method, num_rounds = 5):
    all_pred = np.zeros((num_rounds, y_test.shape[0]))
    rng = np.random.RandomState(123)
    for i in range(num_rounds):
        sample_indices = np.arange(X_train.shape[0])
        bootstrap_indices = rng.choice(sample_indices, size=sample_indices.shape[0], replace=True)
        X_boot = X_train[bootstrap_indices]
        y_boot = y_train[bootstrap_indices]
        model = LogRegModel(X_boot, X_test, y_boot, y_test, alpha, max_iter, method)
        all_pred[i] = model["test_prediction"].reshape(y_test.shape[0],)
    
    main_predictions = stats.mode(all_pred, axis=0).mode
    y_test = y_test.reshape(y_test.shape[0],)
    avg_bias = np.sum(main_predictions != y_test) / y_test.size

    var = np.zeros(all_pred[0].shape)
    for pred in all_pred:
        var += (pred != main_predictions).astype(np.int_)
    var /= num_rounds
    avg_var = var.sum() / y_test.shape[0]

    return avg_bias, avg_var

    
def normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_normalized = (data - mean)/std
    return data_normalized

def computational_cost(X_train, X_test, Y_train, Y_test, alpha, max_iter):
    methods = ["SGD", "ISGD", "ADAM"]
    for method in methods:
        t_start = process_time()
        for i in range(5):
            model = LogRegModel(X_train, X_test, Y_train, Y_test, alpha, max_iter, method) 
        t_end = process_time()
        print('The', method ,'method for MNIST has a computational cost of', (t_end - t_start)/5 ,'seconds.') 
       
def bias_vs_iterations(X_train, X_test, Y_train, Y_test, alpha):
    methods = ["SGD", "ISGD", "ADAM"]
    iterations = [50, 100, 250, 500, 750, 1000]
    bias = [[] for i in range(3)]
    for ind, method in enumerate(methods):
        for max_iter in iterations:
            avg_bias, avg_var = bias_variance_decomp(X_train, Y_train, X_test,Y_test, alpha, max_iter, method, num_rounds = 5)
            bias[ind].append(avg_bias)
            if (max_iter == 1000):
                print('The', method ,'method for MNIST has an asymptotic variance of', avg_var)

    for method in bias:
        plt.plot(iterations, method)
    plt.legend(methods)
    plt.xlabel("Iterations")
    plt.ylabel("Bias")
    sns.despine()
    plt.savefig("./Graphs/MNIST_bias_iter.png")
    plt.show()
    plt.close()

    for method in bias:
        plt.plot(iterations, method)
    plt.legend(methods)
    plt.xlabel("Iterations")
    plt.ylabel("log(Bias)")
    plt.yscale('log')
    sns.despine()
    plt.savefig("./Graphs/MNIST_bias_iter_log.png")
    plt.show()
    plt.close()
    
def bias_vs_alpha(X_train, X_test, Y_train, Y_test, max_iter):
    methods = ["SGD", "ISGD", "ADAM"]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    bias = [[] for i in range(3)]
    for ind, method in enumerate(methods):
        for alpha in alphas:
            avg_bias, avg_var = bias_variance_decomp(X_train, Y_train, X_test,Y_test, alpha, max_iter, method, num_rounds = 5)
            bias[ind].append(avg_bias)

    for method in bias:
        plt.plot(alphas, method)
    plt.legend(methods)
    plt.xlabel("Learning Rates")
    plt.ylabel("Bias")
    sns.despine()
    plt.savefig("./Graphs/MNIST_bias_alpha.png")
    plt.show()
    plt.close()

    for method in bias:
        plt.plot(alphas, method)
    plt.legend(methods)
    plt.xlabel("Learning Rates")
    plt.ylabel("log(Bias)")
    plt.yscale('log')
    sns.despine()
    plt.savefig("./Graphs/MNIST_bias_alpha_log.png")
    plt.show()
    plt.close()


def main():
    mnist = loadmat("./Input/mnist-original.mat")
    mnist_data = mnist["data"].T
    mnist_label = mnist["label"][0]
    mnist_data_normalized = normalize(mnist_data)
    X_train, X_test, Y_train, Y_test = train_test_split(mnist_data_normalized, mnist_label, test_size=0.20, random_state=42)
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    Y_test = Y_test.reshape(Y_test.shape[0],1)

    Y_train_8=(Y_train==8).astype(int)
    Y_test_8=(Y_test==8).astype(int)

    # Experiment 1
    computational_cost(X_train, X_test, Y_train_8, Y_test_8, alpha=0.01, max_iter=1000)

    # Experiment 2 and 4
    bias_vs_iterations(X_train, X_test, Y_train_8, Y_test_8, alpha=0.01)

    # Experiment 3
    bias_vs_alpha(X_train, X_test, Y_train_8, Y_test_8, max_iter=1000)
    
def citation():
    # Source for Logistic Model 
    print("Boulahia, Hamza. “Logistic Regression MNIST Classification.” Kaggle.com, 2020, www.kaggle.com/code/hamzaboulahia/logistic-regression-mnist-classification/notebook.")

    # Source for bias_variance_decomp function:
    print("Raschka, Sebastian. “MLxtend: Providing Machine Learning and Data Science Utilities and Extensions to Python’s Scientific Computing Stack.” The Journal of Open Source Software, vol. 3, no. 24, Apr. 2018, joss.theoj.org/papers/10.21105/joss.00638, https://doi.org/10.21105/joss.00638.")
    print("https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/bias_variance_decomp.py")

    # Source for Dataset
    print("https://www.kaggle.com/datasets/avnishnish/mnist-original/data")


main()