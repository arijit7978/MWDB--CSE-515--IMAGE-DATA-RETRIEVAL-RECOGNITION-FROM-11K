from argparse import ArgumentParser
import sys,os,webbrowser
import pandas as pd
from scipy.spatial import distance_matrix,distance
import numpy as np
import random
from feature_extraction import hog
import time
start_time = time.time()

def max_a_mean(data,c):

    initial_centroids = []
    image_list = data.index
    random_point = image_list[random.randrange(len(image_list))]

    dist_mat = distance_matrix(data.values[:, 1:],data.values[:, 1:])
    max_index = np.argmax(dist_mat[random_point])
    initial_centroids.append(max_index)
    point1 = initial_centroids[0]
    max_index = np.argmax(dist_mat[point1])
    initial_centroids.append(max_index)

    for index in range(2, c):
        max_dist_sum = 0
        for candidate in image_list:
            dist_sum = 0
            for image in initial_centroids:
                dist_sum += dist_mat[image][candidate]
            if dist_sum > max_dist_sum:
                max_index = candidate
                max_dist_sum = dist_sum
        initial_centroids.append(max_index)

    centroid = np.empty([c,len(data.values[0])-1])
    i=0
    for index in initial_centroids:
        centroid[i] = data.values[index,1:]
        i += 1

    return centroid


def kmeans(input,c):
    data = input.values
    # centroids = data[:c,1:]
    centroids = max_a_mean(input,c)

    while True:
        #compute distance matrix
        dist = distance_matrix(data[:,1:],centroids)
        cluster_dict = dict()
        cluster = dist.argmin(axis=1)
        #find group of points in each cluster
        for i in range(len(cluster)):
            temp = data[i]
            temp = temp.reshape(1, len(data[i]))
            if str(cluster[i]) not in cluster_dict:
                cluster_dict[str(cluster[i])] = temp
            else:
                cluster_dict[str(cluster[i])] = np.concatenate((cluster_dict[str(cluster[i])],temp))

        updated_centroid = np.zeros((0,len(data[0])-1))

        #recompute centroid of each cluster
        for key,value in cluster_dict.items():
            mean = np.mean(value[:,1:],axis=0)
            mean = mean.reshape(1,len(mean))

            updated_centroid = np.concatenate((updated_centroid, mean))

        # if old centroids equal new centroids then break
        if np.array_equal(centroids,updated_centroid):
            break
        else:
            centroids = updated_centroid

    return cluster_dict,updated_centroid


def metadata_labels(metadata_file,label):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['gender', 'accessories', 'aspectOfHand', 'imageName'])
    if (label.lower() == 'right-hand'):
        d = df[df['aspectOfHand'].str.contains('right')]
    elif (label.lower() == 'left-hand'):
        d = df[df['aspectOfHand'].str.contains('left')]
    elif (label.lower() == 'dorsal'):
        d = df[df['aspectOfHand'].str.contains('dorsal')]
    elif (label.lower() == 'palmar'):
        d = df[df['aspectOfHand'].str.contains('palmar')]
    elif (label.lower() == 'male'):
        d = df[df['gender'].str.match('male')]
    elif (label.lower() == 'female'):
        d = df[df['gender'].str.match('female')]
    elif (label.lower() == 'with accessories'):
        d = df[df['accessories'].astype(str).str.match('1')]
    elif (label.lower() == 'without accessories'):
        d = df[df['accessories'].astype(str).str.match('0')]
    else:
        d=df
    label_list = d['imageName'].tolist()
    return label_list


def create_html(dorsal,palmar,filename,img_folder):
    html_op = ("<html><head><title>%s</title></head><body><h2><b>Output for Task %d</b></h2>" % (filename,2))
    for key,value in dorsal.items():
        html_op += ("<h3>Label:%s &nbsp; Cluster Id:%s</h3>"%("Dorsal",str(int(key)+1)))
        html_op += ("<table>")
        i=0
        for img in value:
            if i%6 == 0:
                if i != 0:
                    html_op += ("</tr>")
                html_op += ("<tr>")
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (img_folder, img[0],img[0][:-4]))
                i += 1
            else:
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>"%(img_folder, img[0],img[0][:-4]))
                i += 1

        html_op += ("</tr></table>")

    for key, value in palmar.items():
        html_op += ("<h3>Label:%s &nbsp; Cluster Id:%s</h3>" % ("Palmar", str(int(key) + 1)))
        html_op += ("<table>")
        i = 0
        for img in value:
            if i % 6 == 0:
                if i != 0:
                    html_op += ("</tr>")
                html_op += ("<tr>")
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (
                img_folder, img[0], img[0][:-4]))
                i += 1
            else:
                html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s</div></td>" % (
                img_folder, img[0], img[0][:-4]))
                i += 1
        html_op += ("</tr></table>")

    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)
    webbrowser.open('file://' + os.path.realpath(filename))




# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-c", "--cluster_count", type=int, help="Number of clusters")
    parser.add_argument("-lf", "--labelled_folder", type=str, help="labelled image folder path")
    parser.add_argument("-uf", "--unlabelled_folder", type=str, help="unlabelled image folder path")
    parser.add_argument("-lm", "--labelled_metadata", type=str, help="Metadata file")
    parser.add_argument("-l", "--labeled_set", type=int, help="Labeled Set number")
    parser.add_argument("-u", "--unlabeled_set", type=int, help="unLabeled Set number ")
    return parser.parse_args(argv)


# Fetch input paramaters
model='hog'
args = read_argument(sys.argv[1:])
c = args.cluster_count
labelled_folder = args.labelled_folder
unlabelled_folder = args.unlabelled_folder
metadata =  args.labelled_metadata
labeled_set=args.labeled_set
unlabeled_set=args.unlabeled_set


csv_file =model+'_labeled_set'+str(labeled_set)+'.csv'

if not os.path.exists(csv_file):
    hog(labelled_folder,csv_file)
df = pd.read_csv(csv_file, sep=',', header=None)
label_dorsal = metadata_labels(metadata, 'dorsal')
label_palmar = metadata_labels(metadata, 'palmar')

df_dorsal = df.loc[df[0].isin(label_dorsal)]
df_palmar = df.loc[df[0].isin(label_palmar)]
df_dorsal = df_dorsal.reset_index(drop=True)
df_palmar = df_palmar.reset_index(drop=True)

# Compute clusters for dorsal and palmar data
cluster_dorsal,centroid_dorsal = kmeans(df_dorsal,c)
cluster_palmar,centroid_palmar = kmeans(df_palmar,c)

# Cluster details command line and html format

# Compute feature descriptors for unlabelled data
csv_file =model + '_unlabeled_set' + str(unlabeled_set) + '.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)

hog(unlabelled_folder,csv_file)

df = pd.read_csv(csv_file, sep=',', header=None)


dist_dorsal = distance_matrix(df.values[:,1:],centroid_dorsal)
dist_palmar = distance_matrix(df.values[:,1:],centroid_palmar)


total_count = len(df.values)

result=[]
for i in range(len(df.values)):
    if min(dist_dorsal[i]) < min(dist_palmar[i]):
        result.append([df[0][i],"dorsal"])
    else:
        result.append([df[0][i],"palmar"])

def testing_accuracy(result,unlabeled_set):
    positive = 0
    negative = 0
    if unlabeled_set == 1:
        count = 1
        for i in result:
            print(i)
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
    return positive,negative



# print_cluster_details(cluster_dorsal,centroid_dorsal,cluster_palmar,centroid_palmar)
positive,negative=testing_accuracy(result,unlabeled_set)
create_html(cluster_dorsal,cluster_palmar,"task2_output.html",labelled_folder)
print(time.time() - start_time)


