import numpy as np
import time
import os
import pandas as pd
from skimage import feature,io,transform
import csv
import math
from math import *
from sklearn.utils.extmath import randomized_svd
import pickle
import webbrowser
import sys
from argparse import ArgumentParser


# Calculates Denominator for Cosine Similarity
def square_rooted(x):
    return sqrt(sum([a * a for a in x]))

#Cosine Similarity
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)

# Calculate Euclidean distance between two descriptors
def calculate_euclidean_distance(descriptor,test_descriptor):
    distance = math.sqrt(sum([(float(a)-float(b))**2 for a,b in zip(descriptor,test_descriptor)]))
    return distance

# SVD
def svd(svd_matrix):
    u, s, vt = randomized_svd(svd_matrix, n_components=256, n_iter=3, random_state=None)
    return {'u': u, 's': s, 'vt': vt}


def hog(directory):

    filename_list = os.listdir(directory)
    for filename in filename_list:
        img_name = os.path.join(directory, filename)
        img = io.imread(img_name)
        img1 = transform.rescale(img, 0.1)
        fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, visualize=True,
                       feature_vector=True,
                       block_norm="L2-Hys")
        result = [filename] + np.array(fd).tolist()

        # Appending descriptors to csv
        with open('hog_11k.csv', 'a',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(result)
        csvFile.close()



def hog_un(image,directory):
    img = io.imread(os.path.join(directory, image))
    img1 = transform.rescale(img, 0.1)
    fd, hog2 = feature.hog(img1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, visualize=True,feature_vector=True,block_norm="L2-Hys")
    return fd

def create_html(final,filename,t):
    html_op = ("<html><head></head><body><h2><b>Output for Task 5</b></h2>")
    html_op += ("<h2>Overall Images %s </h2>" % (overall_images))
    html_op += ("<h2>Unique Images %s &nbsp; </h2>" % (unique_images))
    html_op += ("<h3>Top %s similar Images &nbsp; </h3>" % (t))
    html_op += ("<table>")

    i = 0
    for image in range(min(len(final),t)):
        if i % 6 == 0:
            html_op += ("<tr>")

        html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>ImageId:%s  Score:%f</div></td>" % (folder_path , final[image][0], final[image][0] , final[image][1]))
        if (i+1)%6==0:
            html_op+=("</tr>")
        i+=1
    html_op += ("</tr></table>")
    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)

# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-l", "--layers", type=int, help="Number of Layers")
    parser.add_argument("-k", "--hashes", type=int, help="Number of Hashes per Layer")
    parser.add_argument("-t", "--similar_image", type=int, help="Number of similar Images")
    parser.add_argument("-q", "--query_image", type=str, help="query folder")
    parser.add_argument("-query", "--query", type=str, help="query")
    parser.add_argument("-folder-path","--folder-path",type=str,help="folder path")
    return parser.parse_args(argv)


start_time = time.time()
model='hog'
technique='svd'
dimension=256
csv_file = model.lower() + '_11k.csv'

# # Fetch input paramaters
args = read_argument(sys.argv[1:])
layers = args.layers
hashes= args.hashes
t=args.similar_image
query_image=args.query_image
folder_path=args.folder_path


if not os.path.exists(model + '_11k_' + technique  + '.pickle'):
    print("Executing HOG")
    if not os.path.exists(csv_file):
        hog(folder_path)

    df = pd.read_csv(csv_file, sep=',', header=None)
    technique_result = svd(df.values[:, 1:].astype(float))
    images=df[0].values

    with open(model + '_11k_' + technique +'.pickle', 'wb') as handle:
        pickle.dump(technique_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(model + '_11k_' + technique +'_images.pickle', 'wb') as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(model + '_11k_' + technique + '.pickle', 'rb') as f:
        technique_result = pickle.load(f)
    with open(model + '_11k_' + technique + '_images.pickle', 'rb') as f:
        images = pickle.load(f)

random_vector = [np.random.randn(hashes, len(technique_result['u'][0])) for j in range(layers)]
features=hog_un(query_image,folder_path)
technique_result_query_image=np.array(features).dot(np.array(technique_result['vt']).T)
technique_result_query_image=technique_result_query_image.reshape((1,256))


relevant_images=[]
for layer in range(layers):
    layered_hash_bucket=dict()
    for i,image in zip(technique_result['u'],images):
        hash_value=random_vector[layer].dot(np.array(i))
        hash_value="".join(['1' if i>0 else '0' for i in hash_value])
        layered_hash_bucket.setdefault(hash_value,[]).append(image)



    query_layered_hash_bucket=dict()
    for i, image in zip(technique_result_query_image, [query_image]):
        hash_value = random_vector[layer].dot(np.array(i))
        hash_value = "".join(['1' if i>0  else '0' for i in hash_value])
        query_layered_hash_bucket.setdefault(hash_value, []).append(image)

    for hash in query_layered_hash_bucket.keys():
        if hash in layered_hash_bucket:
            relevant_images.extend(layered_hash_bucket[hash])

overall_images=len(relevant_images)
relevant_images=list(set(relevant_images))
unique_images=len(relevant_images)

print("Overall Images :", overall_images)
print("Unique Images :", unique_images)

relevant_images_vector=[]
final=[]
for i in relevant_images:
    image_index=list(images).index(i)
    relevant_images_vector.append([i,technique_result['u'][image_index]])
    final.append([i,calculate_euclidean_distance(technique_result['u'][image_index],technique_result_query_image[0])])
final = sorted(final, key=lambda p: p[1])



create_html(final, "task5_output.html",t)


resultant_relevant_images_vector=[]
for i in relevant_images_vector:
    if i[0] in [item[0] for item in final[:t]]:
        resultant_relevant_images_vector.append(i)


print("%s similar images for the query image %s" % (t, query_image))
print(" ")

for i in range(t):
    try:
        image_path = folder_path + final[i][0]
        print("ImageId:%s" % (final[i][0]))
    except:
        break

query_image_result=[query_image,technique_result_query_image]

with open(model + '_11k_' + technique + '_'+str(t)+'_relevant_images.pickle', 'wb') as handle:
    pickle.dump(resultant_relevant_images_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(model + '_11k_' + technique + '_similar_images.pickle', 'wb') as handle:
    pickle.dump(relevant_images_vector, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(model + '_11k_' + technique +'_query_image.pickle', 'wb') as handle:
    pickle.dump(query_image_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Execution time:%s seconds"%(time.time()-start_time))


