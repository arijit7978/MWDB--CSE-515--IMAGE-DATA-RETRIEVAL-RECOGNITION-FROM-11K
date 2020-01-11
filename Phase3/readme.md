## CSE 515: Multimedia and Web Databases

## Phase #3

*******************************************************************************************************************************************************
## Author : Group 10
*******************************************************************************************************************************************************
##Description

In this project, we are experimenting with
* clustering
* indexing
* classification / relevance feedback

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
	->pandas
	->scipy  
	->scikit-image==0.15.0

*******************************************************************************************************************************************************

## Getting Started

## Creating Python virtual environment
	-> Create python virtual environment using below steps
		* Install virtualenv using below command:
		  > sudo apt-get install python3-venv
		
		* Go to desired directory and create a virtual environment using below command:
		  > python3 -m venv phase3env

		* Activate virtual environment using below command:
		  > source phase3env/bin/activate

## Installing Python Libraries(requirements.txt has installation steps for all packages)
	> pip3 install -r requirements.txt

*******************************************************************************************************************************************************

## Steps to execute code

	## Place code, requirements.txt in a folder

	## Open terminal and go to the directory where code is present. Use below commands to execute tasks

	## Task 1:
		Command
			> python3 task1.py -lf <labelled dataset path>  -uf <unlabelled dataset path> -lm <labelled dataset metadata path> -l <labelled set number> -u <Unlabelled set number> -k <latent semantic>

			<labelled dataset path>: The path of the labelled dataset folder
			<unlabelled dataset path>: The path of the unlabelled dataset folder
			<labelled dataset metadata path>: The path of the labelled dataset's metadata
            <labelled set number>: Labelled dataset number 
            <Unlabelled set number>: Unlabelled dataset number
			<latent semantic>: Number of latent semantic(k)

		Example:
			python3 task1.py -lf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Labelled/Set1/" -uf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Unlabelled/Set 1/" -lm "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/labelled_set1.csv" -l 1 -u 1 -k 30
*******************************************************************************************************************************************************

	## Task 2:
		Command
			> python3 task2_kmeans.py -c <Number of clusters> -lf <labelled dataset path>  -uf <unlabelled dataset path> -lm <labelled dataset metadata path> -l <labelled set number> -u <Unlabelled set number>

			<Number of clusters>: Number of clusters to be created
			<labelled dataset path>: The path of the labelled dataset folder
            <unlabelled dataset path>: The path of the unlabelled dataset folder
            <labelled set number>: Labelled dataset number 
            <Unlabelled set number>: Unlabelled dataset number
		Example:
			python3 task2_kmeans.py -c 5 -lf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Labelled/Set2/" -uf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Unlabelled/Set 1/" -lm "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/labelled_set2.csv" -l 2 -u 1
*******************************************************************************************************************************************************

    ## Task 3:
        Command
            > python3 task3.py -k <Outgoing Edges> -K <Most Dominant Images> -f<labelled Folder Path> -q <query images>
        Example:
            > python3 task3.py -k 5 -K 10 -f D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Labelled/Set2/ -q "Hand_0008333.jpg,Hand_0006183.jpg,Hand_0000074.jpg"
 
 
*******************************************************************************************************************************************************
 
    ## Task 4:
        Command
            > python3 task4.py -c <Type of Classifier>  -lf <labelled folder> -uf <unlabelled folder> -lm <labelled images metadata> -l <labelled set number> -u <unlabelled set number>
 
        Example:
            > python task4.py -c svm -lf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Labelled/Set2/" -uf "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/Unlabelled/Set2/" -lm "D:/Masters/FALL-2019/MWDB/Phase3/phase3_sample_data/labelled_set2.csv" -l 2 -u 2
*******************************************************************************************************************************************************
 
    ## Task 5:
        Command
            > python3 task5.py -l<number of layers> -k <number of hashes per layer> -t<Set of vectors> -q <Query Image> -folder-path <path of images folder>
 
        Example:
            > python3 task5.py -l 10 -k 10 -t 20 -q Hand_0000674.jpg -folder-path "Hands/"
*******************************************************************************************************************************************************

    ## Task 6:
        Command
            > python3 task6_relevance_feedback_1pass.py -t <relevance feedback type> -f <Folder of 11k images>

				<relevance feedback type> : Allowed values - probabilistic/decision_tree/ppr/svm
        Example:
            > python3 task6_relevance_feedback_1pass.py -t decision_tree -f /home/mansi/PycharmProjects/mwdb_phase3/Hands/
*******************************************************************************************************************************************************