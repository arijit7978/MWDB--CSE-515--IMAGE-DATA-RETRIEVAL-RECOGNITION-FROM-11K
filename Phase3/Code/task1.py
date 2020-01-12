import os
import pandas as pd
import numpy as np
import math
import pickle
import os
from skimage import feature,io,transform
from sklearn.utils.extmath import randomized_svd
import sys
from argparse import ArgumentParser
import time


def hog(directory):

    print("Executing HOG")
    hog_result = []
    filename_list = os.listdir(directory)
    for filename in filename_list:
        img_name = os.path.join(directory, filename)
        img = io.imread(img_name)
        img1 = transform.rescale(img, 0.1)
        fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                               transform_sqrt=False, visualize=True,
                               feature_vector=True,
                               block_norm="L2-Hys")
        result = [filename] + np.array(fd).tolist()
        hog_result.append(result)
    return hog_result


def calculate_euclidean_distance(descriptor,test_descriptor):
    distance = math.sqrt(sum([(float(a)-float(b))**2 for a,b in zip(descriptor,test_descriptor)]))
    return distance

def hog_un(filename,folder_path):
    img_name = os.path.join(folder_path, filename)
    img = io.imread(img_name)
    # rescale image to 10%
    img1 = transform.rescale(img, 0.1)
    fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, visualize=True,
                   feature_vector=True,
                   block_norm="L2-Hys")
    result = np.array(fd).tolist()
    return [result]

#SVD
def svd(svd_matrix):
    u, s, vt = randomized_svd(svd_matrix, n_components=k, n_iter=3, random_state=None)
    return {'u': u, 's': s, 'vt': vt}


def metadata_labels(metadata_file,label):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['aspectOfHand', 'imageName'])
    if (label.lower() == 'dorsal'):
        d = df[df['aspectOfHand'].str.contains('dorsal')]
    elif (label.lower() == 'palmar'):
        d = df[df['aspectOfHand'].str.contains('palmar')]
    label_list = d['imageName'].tolist()
    return label_list

def labelled_technique_result(df,label):

    if not os.path.exists(model + '_' + technique + '_' + label +'_labeled_set'+str(labeled_set)+ '.pickle'):
        technique_result= svd(df.values[:, 1:].astype(float))


        with open(model + '_' + technique +  '_' + label +'_labeled_set'+str(labeled_set)+ '.pickle', 'wb') as handle:
            pickle.dump(technique_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(model + '_' + technique + '_' + label +'_labeled_set'+str(labeled_set)+ '.pickle', 'rb') as f:
            technique_result = pickle.load(f)

    return technique_result


model='hog'
technique='svd'


def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-k", "--k", type=int, help="Latent_Semantics")
    parser.add_argument("-lf", "--labeled_folder", type=str, help="Labeled Folder path")
    parser.add_argument("-uf", "--unlabeled_folder", type=str, help="Unlabeled Folder path")
    parser.add_argument("-lm", "--labeled_metadata", type=str, help="Labeled metadata")
    parser.add_argument("-l", "--labeled_set", type=int, help="Labeled Set number")
    parser.add_argument("-u", "--unlabeled_set", type=int, help="unLabeled Set number ")
    return parser.parse_args(argv)

start_time=time.time()
args = read_argument(sys.argv[1:])
k=args.k
labeled_folder_path=args.labeled_folder
unlabeled_folder_path=args.unlabeled_folder
labeled_metadata_path=args.labeled_metadata
labeled_set=args.labeled_set
unlabeled_set=args.unlabeled_set


if not os.path.exists(model+'_labeled_set'+str(labeled_set)+'.csv'):
    features = hog(labeled_folder_path)
    features=pd.DataFrame(features)
    features.to_csv(model+'_labeled_set'+str(labeled_set)+'.csv',header=False,index=False)
    print("Done")

else:
    features = pd.read_csv(model+'_labeled_set'+str(labeled_set)+'.csv', sep=',', header=None)

label='dorsal'
dorsal_imgs=metadata_labels(labeled_metadata_path,label)
dorsal_df=features.loc[features[0].isin(dorsal_imgs)]
technique_result_dorsal=labelled_technique_result(dorsal_df,label)

label='palmar'
palmar_imgs=metadata_labels(labeled_metadata_path,label)
palmar_df=features.loc[features[0].isin(dorsal_imgs)]
technique_result_palmar=labelled_technique_result(palmar_df,label)


image_list= os.listdir(unlabeled_folder_path)
result=[]
for i in image_list:
    feat=hog_un(i,unlabeled_folder_path)

    feat = np.array(feat).T
    result_dorsal =np.dot(np.array(technique_result_dorsal['vt']),feat)
    result_palmar =np.dot(np.array(technique_result_palmar['vt']),feat)

    dorsal_dist=0.0
    palmar_dist=0.0
    for j in technique_result_dorsal['u']:
        dorsal_dist+=calculate_euclidean_distance(j,result_dorsal[0])
    for k in technique_result_palmar['u']:
        palmar_dist+=calculate_euclidean_distance(k,result_palmar[0])
    if dorsal_dist <= palmar_dist:
        result.append([i,"dorsal"])
    else:
        result.append([i,"palmar"])

df=pd.DataFrame(result)
for i in range(len(result)):
    print(result[i])
df.to_csv('output_task1_'+'labeled_set'+str(labeled_set)+'_unlabeled_set'+str(unlabeled_set)+'.csv')


positive=0
negative=0
if unlabeled_set==1:
    count=1
    for i in result:
        #print(i,count)
        if count <=50:
            if i[1]=='dorsal':
                positive+=1
            else:
                negative+=1
        else:
            if i[1]=='palmar':
                positive+=1
            else:
                negative+=1
        count+=1


if unlabeled_set==2:
    result_label=['dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal',
        'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal',
        'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar',
        'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar',
        'palmar', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'dorsal', 'dorsal', 'palmar',
        'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'dorsal', 'dorsal', 'palmar',
        'palmar', 'palmar', 'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar',
        'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar',
        'palmar', 'palmar', 'palmar', 'dorsal']
    for i in range(len(result)):
        if result[i][1]==result_label[i]:
            positive+=1
        else:
            negative+=1


#print(positive,negative)
print("Execution time:%s seconds"%(time.time()-start_time))