import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def split_data (dataset):
    """
    Arguments: the diamond's dataset
    Returns: no ruturn
    Save in files the splited dataset
    """
    #1. SPLIT DATASET
    data_size = 53940        #change values here
    train_size = 30000
    validation_size = 10000
    test_size = 13940

    nx = 9                   #number of features of the input
    ny = 1                   #number of fetures of the output

    train,test_validation = train_test_split(dataset, test_size = validation_size + test_size)
    validation, test = train_test_split(test_validation, test_size = test_size)

    print ("data_size: ", data_size)
    print ("train_size: ", train.shape[0])
    print ("validation_size: ", validation.shape[0])
    print ("test_size: ", test.shape[0])


    #2. SAVE THE PANDA'S DATAFRAME ON NUMPY ARRAYS
    categorical_labels = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']

    train_volume = np.empty(train.shape)
    validation_volume = np.empty(validation.shape)
    test_volume = np.empty(test.shape)

    le = LabelEncoder()

    # For each feature, copy or transforms and copy (in categorical case) to the correspondent volume
    for i in range(len(categorical_labels)):
        if( i >= 1 and i <= 3): #if the feature is categorical
            train_volume [:,i] = le.fit_transform(train[categorical_labels[i]])
            validation_volume[:, i] = le.fit_transform(validation[categorical_labels[i]])
            test_volume[:, i] = le.fit_transform(test[categorical_labels[i]])
        else:
            train_volume[:,i] = train[categorical_labels[i]]
            validation_volume[:,i] = validation[categorical_labels[i]]
            test_volume[:,i] = test[categorical_labels[i]]

    # Separates the input from the label organize data in (number_features, number_of_examples)
    x_train = (train_volume[:, 0:9]).T
    y_train = (train_volume[:, 9].reshape(train_size,1)).T
    x_validation = validation_volume [:, 0:9].T
    y_validation = validation_volume [:,9].reshape(ny, validation_size)
    x_test = test_volume [:,0:9].T
    y_test = test_volume [:,9].reshape(test_size,1).T

    np.savetxt('x_train.txt', x_train, delimiter=',')   # X is an array
    np.savetxt('y_train.txt', y_train, delimiter=',')   # X is an array
    np.savetxt('x_validation.txt', x_validation, delimiter=',')   # X is an array
    np.savetxt('y_validation.txt', y_validation, delimiter=',')   # X is an array
    np.savetxt('x_test.txt', x_test, delimiter=',')   # X is an array
    np.savetxt('y_test.txt', y_test, delimiter=',')   # X is an array

    # Sanity check
    assert(x_train.shape == (nx, train_size))
    assert(y_train.shape == (ny, train_size))
    assert(x_validation.shape == (nx, validation_size))
    assert(y_validation.shape == (ny, validation_size))
    assert(x_test.shape == (nx, test_size))
    assert(y_test.shape == (ny, test_size))



def load_vectors():
    x_train = np.loadtxt('x_train.txt', delimiter = ',')
    y_train = np.loadtxt('y_train.txt', delimiter = ',')
    x_validation = np.loadtxt('x_validation.txt', delimiter = ',')
    y_validation = np.loadtxt('y_validation.txt', delimiter = ',')
    x_test = np.loadtxt('x_test.txt', delimiter = ',')
    y_test = np.loadtxt('y_test.txt', delimiter = ',')
    return x_train, y_train, x_validation, y_validation, x_test, y_test