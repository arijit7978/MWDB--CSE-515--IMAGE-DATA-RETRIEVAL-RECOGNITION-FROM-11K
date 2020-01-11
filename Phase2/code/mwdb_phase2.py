import numpy as np
import os,math,sys
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as Lda
from sklearn.decomposition import NMF,TruncatedSVD
import phase1
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
from argparse import ArgumentParser
from math import *
import csv


#PCA
def pca(pca_matrix,k):
    sc = StandardScaler()
    matrix = sc.fit_transform(pca_matrix)
    pca = PCA(n_components=k)
    # fit on data
    u = pca.fit_transform(matrix)
    s = pca.explained_variance_
    vt = pca.components_
    return {'u': u, 's': s, 'vt': vt}

#SVD
def svd(svd_matrix,k):
    u, s, vt = np.linalg.svd(svd_matrix, full_matrices=True)
    return {'u': u[:,:k], 's': s, 'vt': vt[:k]}

#NMF
def nmf(nmf_matrix,k):
    sc = StandardScaler()
    matrix = sc.fit_transform(nmf_matrix)
    model = NMF(n_components=k, init='random', random_state=0)
    min_val = matrix.min()
    if min_val < 0:
        matrix = matrix - min_val
    u = model.fit_transform(matrix)
    vt = model.components_
    return {'u': u, 's': 'None', 'vt': vt}

#LDA
def lda(lda_matrix,k):
    sc = StandardScaler()
    matrix = sc.fit_transform(lda_matrix)
    x = Lda(n_components=k)
    min_val = matrix.min()
    if min_val < 0:
        matrix = matrix - min_val
    u = x.fit_transform(matrix)
    vt = x.components_
    return {'u': u, 's': 'None','vt': vt[:k]}


# Calculate Euclidean distance between two descriptors
def calculate_euclidean_distance(descriptor,test_descriptor):
    distance = math.sqrt(sum([(float(a)-float(b))**2 for a,b in zip(descriptor,test_descriptor)]))
    return distance

# Calculates Denominator for Cosine Similarity
def square_rooted(x):
    return sqrt(sum([a * a for a in x]))

#Cosine Similarity
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return numerator / float(denominator)


def task1(model,technique,dimension,folder_path,label_list):
    # Gets the Latent semantics for the Feature Descriptor from the Phase1
    csv_file = model.lower() + '.csv'
    if not os.path.exists(csv_file):
        exec("%s.%s('%s')"%("phase1",model.lower(),folder_path))
    df = pd.read_csv(csv_file, sep=',', header=None)
    if not label_list==[]:
        df = df.loc[df[0].isin(label_list)]
    x=("%s(%s,%d)"%(technique.lower(),"df.values[:,1:].astype(float)",dimension))
    technique_result=eval(x)
    return technique_result


def extracredit1(model,image_folder,technique_result):
    extra_credit_result={}
    csv_file = model.lower() + '.csv'
    if not os.path.exists(csv_file):
        exec("%s.%s('%s')"%("phase1",model.lower(),image_folder))
    df = pd.read_csv(csv_file, sep=',', header=None)
    for i in range(len(technique_result['u'][0])):
        extra_credit={}
        for j in range(len(technique_result['u'])):
            extra_credit[df.values[j][0]]=technique_result['u'][j][i]
        extra_credit=sorted(extra_credit.items(), key=lambda x: x[1],reverse=True)
        extra_credit_result['k'+str(i+1)]=extra_credit
    return extra_credit_result


def extracredit2(model,image_folder,technique_result):
    extra_credit_result={}
    csv_file = model.lower() + '.csv'
    if not os.path.exists(csv_file):
        exec("%s.%s('%s')"%("phase1",model.lower(),image_folder))
    df = pd.read_csv(csv_file, sep=',', header=None)
    for i in range(len(technique_result['vt'])):
        extra_credit = {}
        dot_product=[]
        for j in range(len(df.values)):
            dot_product.append(sum([x * y for x, y in zip(technique_result['vt'][i],df.values[j][1:].astype(float))]))
        highest_dot_product=max(dot_product)
        extra_credit[df.values[dot_product.index(highest_dot_product)][0]] = highest_dot_product
        extra_credit=sorted(extra_credit.items(), key=lambda x: x[1])
        extra_credit_result['k'+str(i+1)]=extra_credit
    return extra_credit_result


def task2(model,image_folder,image_id,m,task1_u,label_list):
    image_entry=0
    csv_file = model.lower() + '.csv'
    df = pd.read_csv(csv_file, sep=',', header=None)
    if not label_list == []:
        df = df.loc[df[0].isin(label_list)]
    for i in range(len(df.values)):
        if df.values[i][0] == image_id:
            image_entry = i
            break
    img_dist_mapping = dict()
    for row in range(len(task1_u)):
        img_dist_mapping[df.values[row][0]] = calculate_euclidean_distance(task1_u[row],task1_u[image_entry])
    sorted_img=[(entry,img_dist_mapping[entry]) for entry in sorted(img_dist_mapping,key=img_dist_mapping.get)]

    #Calulcating the score
    for i in range(m):
        image_path=image_folder+sorted_img[i][0]
        print("ImageId:%s  Score:%f"%(sorted_img[i][0],sorted_img[i][1]))
        op=Image.open(image_path)
        op.show()

#Meta Data files
def metadata_labels(metadata_file,label):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['gender', 'accessories', 'aspectOfHand', 'imageName'])
    if (label.lower() == 'right'):
        d = df[df['aspectOfHand'].str.contains('right')]
    elif (label.lower() == 'left'):
        d = df[df['aspectOfHand'].str.contains('left')]
    elif (label.lower() == 'dorsal'):
        d = df[df['aspectOfHand'].str.contains('dorsal')]
    elif (label.lower() == 'palmar'):
        d = df[df['aspectOfHand'].str.contains('palmar')]
    elif (label.lower() == 'male'):
        d = df[df['gender'].str.match('male')]
    elif (label.lower() == 'female'):
        d = df[df['gender'].str.match('female')]
    elif (label.lower() == 'withaccessories'):
        d = df[df['accessories'].astype(str).str.match('1')]
    elif (label.lower() == 'withoutaccessories'):
        d = df[df['accessories'].astype(str).str.match('0')]
    else:
        d=df
    label_list = d['imageName'].tolist()
    return label_list

#Labelling the unlabelled images
def task5(model, technique, dimension,metadata_file, image_id,label):
    label_list = metadata_labels(metadata_file, label)
    csv_file = model.lower() + '.csv'
    if not os.path.exists(csv_file):
        exec("%s.%s('%s')" % ("phase1", model.lower(), folder_path))
    df_complete = pd.read_csv(csv_file, sep=',', header=None)
    threshold = 7
    to_be_classified_label_count = 0
    other_label = 0
    if not label_list == []:
        df_label = df_complete.loc[df_complete[0].isin(label_list)]
    if technique == "pca":
        pca = PCA(n_components=dimension)
        pca.fit(df_label.loc[:,1:])
        transformed = pca.transform(df_complete.loc[:,1:])
    elif technique == "svd":
        svd = TruncatedSVD(n_components=dimension)
        svd.fit(df_label.loc[:,1:])
        transformed = svd.transform(df_complete.loc[:, 1:])
    elif technique == "lda":
        lda = Lda(n_components=dimension)
        lda.fit(df_label.loc[:, 1:])
        transformed = lda.transform(df_complete.loc[:, 1:])
    elif technique == "nmf":
        nmf = NMF(n_components=dimension)
        nmf.fit(df_label.loc[:, 1:])
        transformed = nmf.transform(df_complete.loc[:, 1:])

    for i in range(len(df_complete.values)):
        if df_complete.values[i][0] == image_id:
            image_entry = i
            break

    img_dist_mapping = dict()
    for row in range(len(transformed)):
        if image_entry != row:
            img_dist_mapping[df_complete.values[row][0]] = calculate_euclidean_distance(transformed[row],
                                                                                        transformed[image_entry])
    sorted_img = [(entry, img_dist_mapping[entry]) for entry in sorted(img_dist_mapping, key=img_dist_mapping.get)]
    for i in range(threshold):
        if sorted_img[i][0] in label_list:
            to_be_classified_label_count += 1
        else:
            other_label += 1
    if to_be_classified_label_count >= other_label:
        print("System labels image %s as %s" % (image_id, label))
    else:
        print("System does not label image %s as %s" % (image_id, label))


#Computing the similarity matrix for subjects
def similarity(train_dict,metadata_file):
    data = pd.read_csv(metadata_file)
    df = data.groupby('id').apply(lambda x: x.index.tolist())
    subject_matrix = []
    for i in range(len(df.values)):
        subject_list=[]
        for j in range(len(df.values[i])):
            if subject_list ==[]:
                subject_list.extend(train_dict[df.values[i][j]])
            else:
                subject_list = [x + y for x, y in zip(subject_list, train_dict[df.values[i][j]])]
        for k in range(len(subject_list)):
            subject_list[k] /= len(df.values[i])
        subject_matrix.append(subject_list)

    subject_subject_similarity_matrix=[]
    for i in range(len(subject_matrix)):
        subject_subject_similarity_list=[]
        for j in range(len(subject_matrix)):
            distance=cosine_similarity(subject_matrix[i], subject_matrix[j])
            subject_subject_similarity_list.append(distance)
        subject_subject_similarity_matrix.append(subject_subject_similarity_list)
    subjects = list()
    for i in df.index:
        subjects.append(i)
    return subject_subject_similarity_matrix,subjects


# Calculating the top3 similar subjects for the given subject ID
def task6(subject_subject_similarity_matrix,subject_id,subjects):
    length=len(subject_subject_similarity_matrix)
    for i in range(length):
        if subject_id==subjects[i]:
            sim = sorted(list(enumerate(subject_subject_similarity_matrix[i])), key=lambda x: x[1],reverse=True)[1:]
            for i in range(3):
                print("Subject_id:%s  Score:%f"%(subjects[sim[i][0]],sim[i][1]))
            return

# Binary Image metadata computation and performing nmf
def task8(metadata_file):
    data = pd.read_csv(metadata_file)
    df = pd.DataFrame(data, columns=['gender', 'accessories', 'aspectOfHand', 'imageName'])
    r = (df['aspectOfHand'].str.contains('right')) * 1
    l = (df['aspectOfHand'].str.contains('left')) * 1
    d = (df['aspectOfHand'].str.contains('dorsal')) * 1
    p = (df['aspectOfHand'].str.contains('palmar')) * 1
    wa = (df['accessories'].astype(str).str.match('1')) * 1
    wna = (df['accessories'].astype(str).str.match('0')) * 1
    m = (df['gender'].str.match('male')) * 1
    f = (df['gender'].str.match('female')) * 1
    b_mat = {"left-hand": l, "right-hand": r, "dorsal": d, "palmar": p, "with accessories": wa,
             "without accessories": wna, "male": m, "female": f}
    b_mat = pd.DataFrame(b_mat)
    return b_mat

# Creating html for extra credit and Task 1,3
def create_html(mapping_dict,filename,task_id,type,img_folder):
    html_op = ("<html><head><title>%s</title></head><body><h2><b>%s Output for Task %d</b></h2>" % (filename,type,task_id))
    for key,value in mapping_dict.items():
        html_op += ("<h3>Latent Semantic %s</h3>"%(key))
        for img in value:
            html_op += ("<p>Image Id:%s &nbsp; Score:%f</p>"%(img[0],img[1]))
            html_op += ("<img src='%s%s' width=100 height=100><br>"%(img_folder,img[0]))
    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)

#Visualizer for task7
def task_visualizer(technique_result,subjects,task_id):
    extra_credit_result={}
    for i in range(len(technique_result['u'][0])):
        extra_credit={}
        for j in range(len(technique_result['u'])):
            extra_credit[subjects[j]]=technique_result['u'][j][i]
        extra_credit=sorted(extra_credit.items(), key=lambda x: x[1],reverse=True)
        extra_credit_result['k'+str(i+1)]=extra_credit
    filename="datasemantic_task7.html"
    html_op = ("<html><head><title>%s</title></head><body><h2><b>%s Output for Task %d</b></h2>" % (filename,
                                                                                                    "Subject Latent Semantic" ,task_id))
    for key, value in extra_credit_result.items():
        html_op += ("<h3>Subject Latent Semantic%s</h3>" % (key))
        for img in value:
            html_op += ("<p>Subject Id:%s &nbsp; Weight:%f</p>" % (img[0], img[1]))
    html_op += "</body></html>"

    file = open(filename, "w")
    file.write(html_op)

    extra_credit_result = {}
    for i in range(len(technique_result['vt'])):
        extra_credit = {}
        for j in range(len(technique_result['vt'][0])):
            extra_credit[subjects[j]] = technique_result['vt'][i][j]
        extra_credit = sorted(extra_credit.items(), key=lambda x: x[1], reverse=True)
        extra_credit_result['k' + str(i + 1)] = extra_credit
    filename = "featuresemantic_task7.html"
    html_op = ("<html><head><title>%s</title></head><body><h2><b>%s Output for Task %d</b></h2>" % (filename,
                                                                                                    "Latent Semantic Subject",
                                                                                                    task_id))
    for key, value in extra_credit_result.items():
        html_op += ("<h3>Latent Semantic Subject %s</h3>" % (key))
        for img in value:
            html_op += ("<p>Subject Id:%s &nbsp; Weight:%f</p>" % (img[0], img[1]))
    html_op += "</body></html>"

    file = open(filename, "w")
    file.write(html_op)

#Store data into CSV
def csvstore(model,technique,task,technique_result):
    if (os.path.exists(model + '_' + technique + '_' + str(task) + '.csv')):
        os.remove(model + '_' + technique + '_' + str(task) + '.csv')
    x=pd.DataFrame(technique_result['u'])
    x.to_csv(model + '_' + technique + '_' + str(task) + '.csv',header=False, index=False)
    return


# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, help="Dataset folder")
    parser.add_argument("-d", "--metadata", type=str, help="Metadata file")
    parser.add_argument("-m", "--model", type=str, help="Feature extraction model")
    parser.add_argument("-r", "--dimension_reduction", type=str, help="Dimensionality reduction algorithm")
    parser.add_argument("-t", "--task", type=int, help="Task Id")
    parser.add_argument("-i", "--image", type=str, help="Image Id")
    parser.add_argument("-k", "--latent_semantic_count", type=int, help="Latent semantic count")
    parser.add_argument("-s", "--img_count", type=int, help="Similar image count")
    parser.add_argument("-l", "--label", type=str, help="Label")
    parser.add_argument("-u", "--subject_id", type=int, help="Subject Id")
    return parser.parse_args(argv)


args = read_argument(sys.argv[1:])
folder_path = args.folder
metadata_file=args.metadata
technique = args.dimension_reduction
model = args.model
task = args.task
image_id = args.image
label=args.label
dimension=args.latent_semantic_count
m=args.img_count
subject_id=args.subject_id


label_list=[]

if task == 1:
    technique_result = task1(model, technique, dimension, folder_path,label_list)
    csvstore(model, technique, task, technique_result)
    U = extracredit1(model, folder_path,technique_result)
    create_html(U, "datasemantic_task1.html", task, "Data Semantic", folder_path)
    V = extracredit2(model,folder_path, technique_result)
    create_html(V, "featuresemantic_task1.html", task, "Feature Semantic", folder_path)
elif task == 2:
    df = pd.read_csv(model + '_' + technique + '_1' +'.csv', sep=',', header=None)
    task2(model,folder_path, image_id,m,df.values,label_list)

elif task == 3:
    label_list = metadata_labels(metadata_file, label)
    technique_result = task1(model,technique,dimension,folder_path,label_list)
    csvstore(model, technique, task, technique_result)
    U = extracredit1(model, folder_path, technique_result)
    create_html(U, "datasemantic_task3.html", task, "Data Semantic", folder_path)
    V = extracredit2(model, folder_path, technique_result)
    create_html(V, "featuresemantic_task3.html", task, "Feature Semantic", folder_path)
elif task == 4:
    df = pd.read_csv(model + '_' + technique + '_3' + '.csv', sep=',', header=None)
    label_list = metadata_labels(metadata_file, label)
    task2(model, folder_path, image_id, m, df.values,label_list)
elif task == 5:
    task5(model, technique, dimension,metadata_file, image_id, label)
elif task == 6:
    df = pd.read_csv(model + '_' + technique + '_1'  + '.csv', sep=',', header=None)
    subject_subject_similarity_matrix,subjects=similarity(df.values,metadata_file)

    if (os.path.exists('subject_subject_similarity_matrix.csv')):
        os.remove('subject_subject_similarity_matrix.csv')
    x = pd.DataFrame(subject_subject_similarity_matrix)
    x.to_csv('subject_subject_similarity_matrix.csv',header=False, index=False)

    if (os.path.exists('subjects.csv')):
        os.remove('subjects.csv')
    x = pd.DataFrame(subjects)
    x.to_csv('subjects.csv',header=False, index=False)
    task6(subject_subject_similarity_matrix,subject_id,subjects)

elif task == 7:
    df = pd.read_csv('subject_subject_similarity_matrix.csv', sep=',', header=None)
    df2 = pd.read_csv('subjects.csv', sep=',', header=None)
    subjects=[]
    for i in df2.values:
        subjects.append(i[0])
    technique_result = nmf(df.values,dimension)
    task_visualizer(technique_result, subjects,task)

elif task == 8:
    metadata_mat = task8(metadata_file)
    technique_result = nmf(metadata_mat, dimension)
    U = extracredit1(model, folder_path, technique_result)
    create_html(U, "datasemantic_task8.html", task, "Image Space Data Semantic", folder_path)
    V = extracredit2(model, folder_path, technique_result)
    create_html(V, "featuresemantic_task8.html", task, "Metadata Space Feature Semantic", folder_path)

