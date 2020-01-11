import os
import pandas as pd
import numpy as np
from math import *
from PIL import Image
from skimage import feature


def lbp(directory):

    window_width = 100
    window_height = 100

    lbp= list()
    for imagename in os.listdir(directory):
        im = Image.open(directory + imagename).convert('L')
        imagewidth, imageheight = im.size
        lbp_feature_vector = list()
        lbp_feature_vector.append(imagename)
        for i in range(0, imageheight, window_height):
            for j in range(0, imagewidth, window_width):
                box = (j, i, j + window_width, i + window_height)
                a = im.crop(box)
                lbp_features = feature.local_binary_pattern(a, 24, 3, method="uniform")
                (hist, _) = np.histogram(lbp_features.ravel(), density=True, bins=12)
                lbp_feature_vector.extend(hist)
        lbp.append(lbp_feature_vector)
    return lbp


# Calculates Denominator for Cosine Similarity
def square_rooted(x):
    return sqrt(sum([a * a for a in x]))


# Cosine Similarity
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)


def metadata_labels(metadata_file,label):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['aspectOfHand', 'imageName'])
    if (label.lower() == 'dorsal'):
        d = df[df['aspectOfHand'].str.contains('dorsal')]
    elif (label.lower() == 'palmar'):
        d = df[df['aspectOfHand'].str.contains('palmar')]
    label_list = d['imageName'].tolist()
    return label_list



def PPR(combined_df,labeled_images):
    feature_vector=combined_df.values[:, 1:]
    random_walk = list()
    for i in range(len(feature_vector)):
        similar_images = []
        for j in range(len(feature_vector)):
            if i == j:
                similar_images.append(0)
            else:
                similar_images.append(cosine_similarity(feature_vector[i],feature_vector[j]))
        similar_images_copy = sorted(similar_images, reverse=True)[:80]
        for i in range(len(similar_images)):
            if similar_images[i] in similar_images_copy:
                similar_images_copy.remove(similar_images[i])
            else:
                similar_images[i] = 0.0
        similar_images = [(alpha * i) / sum(similar_images) for i in similar_images]
        random_walk.append(similar_images)
    random_walk = np.array(list(map(list, zip(*random_walk))))


    Identity = np.identity(len(random_walk), dtype=float)
    SteadyState = np.linalg.inv(Identity - random_walk)


    seed = [[0.0 for j in range(1)] for i in range(len(feature_vector))]

    for i in range(len(labeled_images)):
        seed[combined_df.index[combined_df[0] == labeled_images[i]].values[0]][0] = (1 - alpha)  / len(labeled_images)
    seed = np.array(seed)
    Eigenvector = SteadyState.dot(seed)
    return Eigenvector


def task6_ppr(task6_input,feedback_relevant,feedback_irrelevant):
    final = list()
    for i in range(len(task6_input)):
        temp = list()
        temp.append(task6_input[i][0])
        temp1 = list(task6_input[i][1])
        temp.extend(temp1)
        final.append(temp)
    df = pd.DataFrame(final)
    relevant_ppr = labeled_PPR(df,feedback_relevant)
    irrelevant_ppr = labeled_PPR(df,feedback_irrelevant)
    return labelling_unlabeled_data(relevant_ppr, irrelevant_ppr,df[0])


def labeled_PPR(combined_df,labeled_imgs):
    x = pd.DataFrame(PPR(combined_df, labeled_imgs))
    x.insert(0, "Images", combined_df[0])
    return list(sorted(x.values,key=lambda p:p[1],reverse=True))


def labelling_unlabeled_data(dorsal_ppr,palmar_ppr,images):
    result = []
    for image in images:
        dorsalindex = -1
        palmarindex = -1
        for j in range(len(dorsal_ppr)):
            if dorsal_ppr[j][0] == image:
                dorsalindex = j
                break

        for j in range(len(palmar_ppr)):
            if palmar_ppr[j][0] == image:
                palmarindex = j
                break

        if dorsalindex < palmarindex:
            result.append('relevant')
        elif dorsalindex == palmarindex:
            result.append(
                ('relevant' if dorsal_ppr[dorsalindex][1] > palmar_ppr[palmarindex][1] else 'irrelevant'))
        else:
            result.append('irrelevant')

    return result



def testing_accuracy(result,unlabeled_set):
    positive = 0
    negative = 0
    if unlabeled_set == 1:
        count = 1
        for i in result:
            print(i, count)
            if count <= 50:
                if i[1] == 'dorsal':
                    positive += 1
                else:
                    negative += 1
            else:
                if i[1] == 'palmar':
                    positive += 1
                else:
                    negative += 1
            count += 1

    if unlabeled_set == 2:
        result_label = ['dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal', 'dorsal', 'dorsal',
                        'dorsal', 'dorsal', 'dorsal',
                        'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal', 'dorsal', 'palmar', 'palmar', 'dorsal',
                        'dorsal', 'dorsal', 'dorsal',
                        'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar',
                        'palmar', 'palmar', 'palmar',
                        'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar',
                        'palmar', 'palmar', 'palmar',
                        'palmar', 'dorsal', 'dorsal', 'dorsal', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar',
                        'dorsal', 'dorsal', 'palmar',
                        'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar', 'palmar',
                        'dorsal', 'dorsal', 'palmar',
                        'palmar', 'palmar', 'palmar', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar',
                        'palmar', 'palmar', 'palmar',
                        'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'dorsal', 'palmar',
                        'palmar', 'palmar', 'palmar',
                        'palmar', 'palmar', 'palmar', 'dorsal']
        for i in range(len(result)):
            if result[i][1] == result_label[i]:
                positive += 1
            else:
                negative += 1

    print(positive, negative)


model='lbp'
alpha=0.85



def main(labeled_folder_path,unlabeled_folder_path,labeled_metadata_path,labeled_set,unlabeled_set):
    labeled_folder_path = labeled_folder_path
    unlabeled_folder_path = unlabeled_folder_path
    labeled_metadata_path =labeled_metadata_path
    labeled_set=labeled_set
    unlabeled_set=unlabeled_set

    print("Executing PPR")
    if not os.path.exists(model+'_labeled_set'+str(labeled_set)+'.csv'):
        features = lbp(labeled_folder_path)
        features=pd.DataFrame(features)
        features.to_csv(model+'_labeled_set'+str(labeled_set)+'.csv',header=False,index=False)

    else:
        features = pd.read_csv(model+'_labeled_set'+str(labeled_set)+'.csv', sep=',', header=None)

    if not os.path.exists(model+'_unlabeled_set'+str(unlabeled_set)+'.csv'):
        unlabeled_features=lbp(unlabeled_folder_path)
        unlabeled_features = pd.DataFrame(unlabeled_features)
        unlabeled_features.to_csv(model + '_unlabeled_set' + str(unlabeled_set) + '.csv',header=False,index=False)

    else:
        unlabeled_features=pd.read_csv(model + '_unlabeled_set' + str(unlabeled_set) + '.csv', sep=',', header=None)

    combined_features = pd.concat([features, unlabeled_features])
    combined_df = combined_features.reset_index(drop=True)

    dorsal_ppr = labeled_PPR(combined_df,labeled_metadata_path, 'dorsal')
    palmar_ppr = labeled_PPR(combined_df,labeled_metadata_path, 'palmar')


    result=labelling_unlabeled_data(dorsal_ppr,palmar_ppr,unlabeled_folder_path)
    # print(result)
    df = pd.DataFrame(result)
    df.to_csv('output_task4_' + 'labeled_set' + str(labeled_set) + '_unlabeled_set' + str(unlabeled_set) + '.csv')
    testing_accuracy(result,unlabeled_set)










