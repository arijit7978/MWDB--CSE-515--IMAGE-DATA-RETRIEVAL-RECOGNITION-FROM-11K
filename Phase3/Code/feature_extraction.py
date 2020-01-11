import csv,os
from skimage import feature,io,transform
import numpy as np
from PIL import Image
import cv2
import scipy


def hog(directory,csv_file):
    '''
    :param directory: (string)Image dataset folder path
    '''

    # Check if csv exists,if so delete the csv
    if (os.path.exists(csv_file)):
        os.remove(csv_file)

    filename_list = os.listdir(directory)

    # loop over all images in specified folder path
    for filename in filename_list:
        img_name = os.path.join(directory, filename)
        img = io.imread(img_name)
        # rescale image to 10%
        img1 = transform.rescale(img, 0.1)
        # fetch feature descriptor using hog for each image
        # number of orientation bins = 9,
        # cell size = 8,
        # block size = 2,
        # L2 - norm clipping threshold = 0.2
        fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, visualize=True,
                       feature_vector=True,
                       block_norm="L2-Hys")
        result = [filename] + np.array(fd).tolist()

        # Appending descriptors to csv
        with open(csv_file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(result)
        csvFile.close()


def cm(directory,csv_file):
    '''

    :param directory:
    :return:
    '''

    # check if csv exists ,delete if exists
    # if (os.path.exists('CMFeature.csv')):
    #     os.remove('CMFeature.csv')

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
        with open(csv_file, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(matrix)


def lbp(directory,csv_file):
    '''
    Input directory
    Computes the lbp features for all the images in the folder.
    Assumes the window_height as 100, window_width as 100, radius as 3 , number_of_points=24 (8*radius) , number_of_bins=10
    :return: None
    '''

    window_width = 100
    window_height = 100
    radius = 3
    number_of_points = 24
    number_of_bins = 10

    #check if csv exists ,delete if exists
    # if (os.path.exists('LBPFeature.csv')):
    #     os.remove('LBPFeature.csv')

    # iterate through a particular folder over all the images
    for imagename in os.listdir(directory):

        # open an image and convert to gray scale
        im = Image.open(directory + imagename).convert('L')

        # compute the width and height
        imagewidth, imageheight = im.size
        lbp_feature_vector = []

        lbp_feature_vector.append(imagename)
        # iterate over all the  windows in the image
        for i in range(0, imageheight, window_height):
            for j in range(0, imagewidth, window_width):

                # select a particular window and compute the lbp features of the window
                box = (j, i, j + window_width, i + window_height)
                a = im.crop(box)

                # lbp features are computed
                lbp_features = feature.local_binary_pattern(a, number_of_points, radius, method="uniform")

                # Normalized using histogram
                (hist, _) = np.histogram(lbp_features.ravel(), density=True, bins=number_of_bins)
                lbp_feature_vector.extend(hist)

        # Append the features to a csv
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(lbp_feature_vector)
        f.close()
