from __future__ import print_function
from PIL import Image
import pandas as pd
import cv2
import scipy
from scipy import stats
import os
import csv
import numpy as np
from math import exp
from math import log
import time
from skimage import feature,io,transform

class Tree:
    def __init__(self,column, value,right_tree,left_tree):
        self.column = column
        self.value = value
        self.right_tree = right_tree
        self.left_tree = left_tree

def divide_tree(datas, col, val):
    tree_values = [[],[]]
    for data in datas:
        if data[col] >= val:
            tree_values[0].append(data)
        else:
            tree_values[1].append(data)
    return tree_values

def current_gini(datas):
    sum_lbl = {}
    for data in datas:
        if data[-1] not in sum_lbl:
            sum_lbl[data[-1]] = 0
        sum_lbl[data[-1]] += 1
    impurity = 1
    for label in sum_lbl:
        impurity -= (sum_lbl[label] / float(len(datas)))**2
    return impurity

def information_gain(left, right, impurity):
    p = float(len(left)) / (len(left) + len(right))
    gin = p * current_gini(left) + (1 - p) * current_gini(right)
    new_impurity = impurity - gin
    return new_impurity

def build_tree(rows):
    best_gain = 0
    best_col = 0
    best_val = 0
    impurity = current_gini(rows)

    for col in range(len(rows[0]) - 1):
        for val in set([row[col] for row in rows]):
            tree_values = divide_tree(rows, col, val)
            left_tree_values, right_tree_values = tree_values[0], tree_values[1]
            if right_tree_values and left_tree_values:
                gain = information_gain(left_tree_values, right_tree_values, impurity)
                if gain >= best_gain:
                    best_gain, best_col, best_val = gain, col, val

    if best_gain >= 0.1:
        tree_values = divide_tree(rows, best_col, best_val)
        right_tree_values, left_tree_values = tree_values[0], tree_values[1]
        right_tree = build_tree(right_tree_values)
        left_tree = build_tree(left_tree_values)
        node = Tree(best_col, best_val, right_tree, left_tree)
        return node
    else:
        return rows[0][-1]

def predict_dcsn(row, node):
    if type(node) == str:
        return node
    elif row[node.column] > node.value:
        return predict_dcsn(row, node.right_tree)
    else:
        return predict_dcsn(row, node.left_tree)

def cm(directory,str):
    '''

    :param directory:
    :return:
    '''

    # check if csv exists ,delete if exists
    if (os.path.exists(str + '.csv')):
        os.remove(str + '.csv')

    height = 100
    width = 100
    for imagename in os.listdir(directory):
        im = Image.open(directory + imagename)
        imgwidth, imgheight = im.size
        imgUMat = cv2.imread(directory + imagename)
        img_yuv = cv2.cvtColor(imgUMat, cv2.COLOR_BGR2YUV)
        matrix = []
        matrix.append(imagename)
        for i in range(0, imgheight, height):
            for j in range(0, imgwidth, width):
                crop_img = img_yuv[i:i + width, j:j + width]
                a = np.array(np.mean(crop_img, axis=(0, 1)))
                b = np.array(np.var(crop_img, axis=(0, 1)))
                c = np.array(scipy.stats.skew(crop_img.reshape(-1, 3)))
                d = np.concatenate((np.concatenate((a, b), axis=0), c), axis=0)
                matrix.extend(d.tolist())
        with open(str + '.csv', 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(matrix)

def convert_to_train_data(df,train_data):
    for i in range(len(train_data)):
        d = df[df['imageName'].str.contains(train_data[i][0])]
        if not (d[d['aspectOfHand'].str.contains("dorsal")]).empty:
            train_data[i][-1] = "dorsal"
        else:
            train_data[i][-1] = "palmar"
        del train_data[i][0]
    return train_data

def train_test_data_dcsn(train_data,test_data):
    # cm("C:\\Users\\Lenovo\\Desktop\\MWDB\\phase3_sample_data\\Unlabelled\\Set 2\\", 'cm_set2_unlabelled')
    # cm("C:\\Users\\Lenovo\\Desktop\\MWDB\\phase3_sample_data\\Labelled\\Set2\\", 'cm_set2')
    # if (os.path.exists('train_data.csv')):
    #     os.remove('train_data.csv')
    #
    # # train_data on features
    #
    # data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\MWDB\\phase3_sample_data\\labelled_set2.csv")
    # df = pd.DataFrame(data, columns=['aspectOfHand', 'imageName'])
    # dfdf = pd.read_csv('cm_set2.csv', sep=',', header=None)
    # ls = dfdf.values[:, 1:]
    # lbp_data = []
    # for i in range(len(dfdf)):
    #     lbp_data.append([dfdf.iloc[i][0]] + ls[i].tolist() + [0])
    # train_data = convert_to_train_data(df,lbp_data)
    # df = pd.DataFrame(train_data)
    # df.to_csv('train_data.csv', index=False,header=None)

    # training_data = pd.read_csv("train_data.csv", header=None)
    # train_data = []
    # for i in range(len(training_data)):
    #     train_data.append(list(training_data.iloc[i]))
    #
    # testing_data = pd.read_csv("cm_set2_unlabelled.csv", header=None)
    # test_data = []
    # for i in range(len(testing_data)):
    #     test_data.append(list(testing_data.iloc[i]))

    labels = list()
    my_tree = build_tree(train_data)
    for row in test_data:
        prediction = predict_dcsn(row[1:], my_tree)
        labels.append(prediction)
        # print(prediction)
    return labels

# Function to return columnwise max-min findMaxMin for scaling
def findMaxMin(x):
    cols = list(zip(*x))
    stats = []
    for e in cols:
        stats.append([min(e), max(e)])
    return stats


# Function to scale the features
def scale(x, stat):
    for row in x:
        for i in range(len(row)):
            row[i] = (row[i] - stat[i][0]) / (stat[i][1] - stat[i][0])


# Function to convert different classes into different columns to implement one v/s all
def multiClass(s):
    m = list(set(s))
    m.sort()
    for i in range(len(s)):
        new = [0] * len(m)
        new[m.index(s[i])] = 1
        s[i] = new
    return m


# Function to compute Theta transpose x Feature Vector
def ThetaTX(Q, X):
    det = 0.0
    for i in range(len(Q)):
        det += X[i] * Q[i]
    return det


# Function to compute cost for negative class (classs = 0)
def LinearSVM_cost0(z):
    if (z < -1):  # Ensuring margin
        return 0
    return z + 1


# Function to compute cost for positive class (classs = 1)
def LinearSVM_cost1(z):
    if (z > 1):  # Ensuring margin
        return 0
    return -z + 1


# function to calculate sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))


# Function to calculate SVM cost
def cost(theta, c, x, y):
    cost = 0.0
    for i in range(len(x)):
        z = ThetaTX(theta[c], x[i])
        # cost += y[i]*LinearSVM_cost1(z) + (1 - y[i])*LinearSVM_cost0(z)
        cost += -1 * (y[i] * log(sigmoid(z)) + (1 - y[i]) * log(1 - sigmoid(z)))
    return cost


# Function to perform Gradient Descent on the weights/parameters
def gradientDescent(theta, c, x, y, learning_rate):
    oldTheta = theta[c]
    for Q in range(len(theta[c])):
        derivative_sum = 0
        for i in range(len(x)):
            derivative_sum += (sigmoid(ThetaTX(oldTheta, x[i])) - y[i]) * x[i][Q]
        theta[c][Q] -= learning_rate * derivative_sum


# Function to return predictions using trained weights
def predict(data, theta):
    predictions = []
    count = 1
    for row in data:
        hypothesis = []
        multiclass_ans = [0] * len(theta)
        for c in range(len(theta)):
            z = ThetaTX(row, theta[c])
            hypothesis.append(sigmoid(z))
        index = hypothesis.index(max(hypothesis))
        multiclass_ans[index] = 1
        predictions.append(multiclass_ans)
        count += 1
    return predictions


# Function to perform cross validation
def cross_validation(x, y, xtest, learning_rate, epoch, validations, labels):
    print("No. of validation checks to be performed: ", validations)
    print("No. of Iterations per validation: ", epoch)
    # accuracies = []
    classification = []
    for valid in range(validations):
        print("\nRunning Validation", valid + 1)

        x_train = x.tolist()
        y_train = y.tolist()
        x_test = xtest.tolist()

        classes = []
        for i in range(len(labels)):
            classes.append([row[i] for row in y_train])
        # Initialising Theta (Weights)
        theta = [[0] * len(x_train[0]) for _ in range(len(classes))]
        # Training the model
        for i in range(epoch):
            for class_type in range(len(classes)):
                gradientDescent(theta, class_type, x_train, classes[class_type], learning_rate)
            if (i % (epoch / 10) == 0):
                print("Processed", i * 100 / epoch, "%")
        print("Completed")
        # Predicting using test data
        y_pred = predict(x_test, theta)

        for item in y_pred:
            if item == [1, 0]:
                # print("dorsal")
                classification.append("relevant")
            else:
                # print("palmar")
                classification.append("irrelevant")
    return classification


def svm(x,y,x_test):
    # Dataset url to be imported
    start = time.time()
    # url = "train_data(set2).csv"
    # dataset = pd.read_csv(url)
    # data = dataset.values
    #
    # url_test = "test_data(unlabelled_set2).csv"
    # dataset_test = pd.read_csv(url_test)
    # data_test = dataset_test.values

    # x = data.values[:, :-1]
    # y = data[:, len(data[0])]
    #
    # x_test = data_test[:, :19]
    stats = findMaxMin(x)
    scale(x, stats)
    # Converting different labels to columns
    # labels can be used later to retrieve the predicted class label in the original form (string format)
    labels = multiClass(y)
    # labels_test = multiClass(y_test)
    # Splitting dataset into training and testing data
    learning_rate = 0.01
    epoch = 400
    validations = 1

    # classification = cross_validation(x, y, x_test, y_test, learning_rate, epoch, validations, labels)
    classification = cross_validation(x, y, x_test, learning_rate, epoch, validations, labels)
    # Printing Final Stats
    return classification
    print("\nReport:")
    print("Model used: ", "Linear SVM using Gradient Descent")
    print("Learning rate: ", learning_rate)
    print("No. of iterations: ", epoch)
    print("No. of validation tests performed: ", validations)
    # print("Accuracy: ",final_score*100,"%")
    print("time(seconds)= ", time.time() - start)
    testing_data = pd.read_csv("hog_set2_unlabelled.csv", header=None)
    test_data = []
    for i in range(len(testing_data)):
        test_data.append(list(testing_data.iloc[i]))

    for i in range(len(classification)):
        print(test_data[i][0],classification[i])

def hog(directory, str):
    '''
    :param directory: (string)Image dataset folder path
    '''

    # Check if csv exists,if so delete the csv
    if (os.path.exists(str + '.csv')):
        os.remove(str + '.csv')

    filename_list = os.listdir(directory)

    # loop over all images in specified folder path
    for filename in filename_list:
        img_name = os.path.join(directory, filename)
        img = io.imread(img_name)
        # rescale image to 10%
        img1 = transform.rescale(img, 0.1)
        fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, visualize=True,
                       feature_vector=True,
                       block_norm="L2-Hys")
        result = [filename] + np.array(fd).tolist()

        # Appending descriptors to csv
        with open(str + '.csv', 'a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(result)
        csvFile.close()
    print("Output CSV File: %s" % (os.getcwd() + '/' + 'HOG.csv'))

def svd(svd_matrix,k):
    u, s, vt = np.linalg.svd(svd_matrix, full_matrices=True)
    return u[:,:k]

def train_test_data_svm():
    # test data on latent semantics
    dfdf = pd.read_csv('hog_unlabelled.csv', sep=',', header=None)
    latent_semantics = svd(dfdf.values[:, 1:].astype(float), 20)

    df = pd.DataFrame(latent_semantics)
    df.to_csv('test_data(unlabelled_set2).csv', index=False,header=None)

    # train_data on latent  semantics
    data = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\MWDB\\phase3_sample_data\\labelled_set2.csv")
    df = pd.DataFrame(data, columns=['aspectOfHand', 'imageName'])
    dfdf = pd.read_csv('hog_set2.csv', sep=',', header=None)
    latent_semantics = svd(dfdf.values[:, 1:].astype(float), 20)

    lbp_data = []
    for i in range(len(dfdf)):
        lbp_data.append([dfdf.iloc[i][0]] + latent_semantics[i].tolist() + [0])
    train_data = convert_to_train_data(df,lbp_data)
    df = pd.DataFrame(train_data)
    df.to_csv('train_data(set2).csv', index=False,header=None)


if __name__ == '__main__':
    chr = input("svm/dcsn: ")
    if chr == "svm":
        train_test_data_svm()
        svm()
    elif chr == "dcsn":
        train_test_data_dcsn()


