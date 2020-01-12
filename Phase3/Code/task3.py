import pandas as pd
from PIL import Image
import os
import pickle
import phase1
from math import *
import numpy as np
from sklearn.utils.extmath import randomized_svd
import webbrowser
from argparse import ArgumentParser
import sys
import time


#SVD
def svd(svd_matrix):
    u, s, vt = randomized_svd(svd_matrix,n_components=dimension,n_iter=5,random_state=None)
    print(u)
    return {'u': u, 's': s, 'vt': vt}



# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-k", "--similar_images", type=int, help="k similar images")
    parser.add_argument("-K", "--dominant_images", type=int, help="K dominant Images")
    parser.add_argument("-f", "--folder_path", type=str, help="Unlabelled Folder Path")
    parser.add_argument("-q", "--ImageIDS", type=str, help="3 Image IDs")
    parser.add_argument("-query", "--query", type=int, help="3 Image IDs")
    return parser.parse_args(argv)

start_time = time.time()
model='hog'
technique='svd'
dimension=30
alpha=0.85

args = read_argument(sys.argv[1:])
k= args.similar_images
K= args.dominant_images
folder_path=args.folder_path
ImageIDS=list(args.ImageIDS.split(','))
query=args.query

csv_file = model.lower() + '.csv'


technique_result={}
if not os.path.exists(model + '_' + technique + '.pickle'):

    if not os.path.exists(csv_file):
        exec("%s.%s('%s')" % ("phase1", model.lower(), folder_path))
    df = pd.read_csv(csv_file, sep=',', header=None)

    x = ("%s(%s)" % (technique.lower(), "df.values[:,1:].astype(float)"))
    technique_result = eval(x)

    with open(model + '_' + technique + '.pickle', 'wb') as handle:
        pickle.dump(technique_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    df = pd.read_csv(csv_file, sep=',', header=None)
    with open(model + '_' + technique + '.pickle', 'rb') as f:
        technique_result = pickle.load(f)



u=technique_result['u']

# Calculates Denominator for Cosine Similarity
def square_rooted(x):
    return sqrt(sum([a * a for a in x]))

#Cosine Similarity
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)

result=[]
for i in range(len(u)):
    temp=[]
    for j in range(len(u)):
        if i==j:
            temp.append(0)
        else:
            temp.append(cosine_similarity(u[i],u[j]))
    temp2=sorted(temp,reverse=True)[:k]
    for i in range(len(temp)):
        if temp[i] in temp2:
            temp2.remove(temp[i])
        else:
            temp[i]=0.0
    temp = [(alpha*i) / sum(temp) for i in temp]
    result.append(temp)

result=list(map(list, zip(*result)))
result=np.array(result)

I=np.identity(len(result),dtype = float)
PPR=I-result

SteadyState=np.linalg.inv(PPR)

seed=[[0.0 for j in range(1)] for i in range(len(u))]
seed[df.index[df[0]==ImageIDS[0]].values[0]][0]=(1-alpha)*1/3
seed[df.index[df[0]==ImageIDS[1]].values[0]][0]=(1-alpha)*1/3
seed[df.index[df[0]==ImageIDS[2]].values[0]][0]=(1-alpha)*1/3
seed=np.array(seed)
Eigenvector=SteadyState.dot(seed)
print(Eigenvector)
# for i in Eigenvector:
#     print(i)

x = pd.DataFrame(Eigenvector)
x.insert(0,"Images",df[0])
for i in x:
    print(i)
# print(x)
# exit(1)
final=sorted(x.values,key=lambda p:p[1],reverse=True)


def create_html(final,filename,K):
    html_op = ("<html><head></head><body><h2><b>Output for Task 3</b></h2>")
    html_op += ("<h3>Top %s similar Images &nbsp; </h3>" % (K))
    html_op += ("<table>")

    i = 0
    for image in range(K):
        if i % 6 == 0:
            html_op += ("<tr>")

        print(folder_path + final[image][0])
        html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>ImageId:%s  Score:%f</div></td>" % (folder_path, final[image][0], final[image][0] , final[image][1]))
        if (i+1)%6==0:
            html_op+=("</tr>")
        i+=1
    html_op += ("</tr></table>")
    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)

create_html(final, "task3_output.html",K)
print("Execution time:%s seconds" % (time.time() - start_time))
