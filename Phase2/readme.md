## CSE 515: Multimedia and Web Databases

## Phase #2

*******************************************************************************************************************************************************
## Author : Group 10
*******************************************************************************************************************************************************
##Description

In this project, we are experimenting with
* image features
* vector models
* dimesionality reduction

Dataset being used is provided by following publication:
Mahmoud Afifi. “11K Hands: Gender recognition and biometric identification using a large dataset of handimages.” M. Multimed Tools Appl (2019) 78: 20835.

CSV file is used to store and retrieve data, Python is used as programming language.
*******************************************************************************************************************************************************

## Pre-requisites:

* Install below software/tools as a pre-requisite of this phase

* Operating system
	-> Ubuntu 18.04

* Programming language
	->Python: 3.7.3

* Python Libraries
	->scikit-image: 0.15.0
	->Pillow: 6.1.0
	->numpy: 1.17.1
	->opencv-python: 3.4.2.16
	->opencv-contrib-python: 3.4.2.16
	->sklearn: 0.0

*******************************************************************************************************************************************************

## Getting Started

## Creating Python virtual environment
	-> Create python virtual environment using below steps
		* Install virtualenv using below command:
		  > sudo apt-get install python3-venv
		
		* Go to desired directory and create a virtual environment using below command:
		  > python3 -m venv phase1env

		* Activate virtual environment using below command:
		  > source phase1env/bin/activate

## Installing Python Libraries(requirements.txt has installation steps for all packages)
	> pip3 install -r requirements.txt

*******************************************************************************************************************************************************

## Steps to execute code

	## Place code, requirements.txt in a folder

	## Open terminal and go to the directory where code is present. Use below commands to execute tasks

	## Task 1:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -k <latent semantic> -t 1
			<dataset path>: the path of dataset folder
			<feature model>: acceptable values sift/hog/cm/lbp
			<dimensionality reduction technique>: pca/svd/lda/nmf
			<latent semantic>: number of latent semantic(k)

		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -m hog -r pca -k 2 -t 1
*******************************************************************************************************************************************************

	## Task 2:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -t 2 -i <test image> -s <similar image count>
			<test image>: the test image
			<similar image count>: count of similar images(m)
		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -m hog -r pca -t 2 -i Hand_0000185.jpg -s 3
*******************************************************************************************************************************************************

	## Task 3:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -k <latent semantic> -t 3 -l <label>
			<label>: acceptable values right/left/dorsal/palmar/male/female/withaccessories/withoutaccessories
		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/	newHandInfo.csv -m hog -r pca -k 2 -t 3 -l female
*******************************************************************************************************************************************************

	## Task 4:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -t 4 -l <label> -i <test image> -s <similar image count>

		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/newHandInfo.csv -m hog -r pca -t 4 -l female -i Hand_0011599.jpg -s 3
*******************************************************************************************************************************************************

	## Task 5:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -k <latent semantic> -t 5 -l <label> -i <test image>

		Example:
			> python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/newHandInfo.csv -m hog -r pca -k 2 -t 5 -l left -i Hand_0011599.jpg
*******************************************************************************************************************************************************

	## Task 6:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -k <latent semantic> -t 6 -u <subject id>
			<subject id>: is the id of the subject (int)
		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/newHandInfo.csv -m hog -r pca -k 2 -t 6 -u 505
*******************************************************************************************************************************************************

	## Task 7:
		Command
			> python3 task1.py -f <dataset path>  -m <feature model> -r <dimensionality reduction technique> -k <latent semantic> -t 7

		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/newHandInfo.csv -m hog -r nmf -k 2 -t 7
*******************************************************************************************************************************************************

	## Task 8:
		Command
			> python3 task1.py -f <dataset path> -r <dimensionality reduction technique> -k <latent semantic> -t 8

		Example:
			python3 task1.py -f '/home/mansi/Documents/sem1/MWDB/phase/phase2/Dataset2/'  -d /home/mansi/Documents/sem1/MWDB/phase/phase2/newHandInfo.csv -r nmf -k 2 -t 8
*******************************************************************************************************************************************************



