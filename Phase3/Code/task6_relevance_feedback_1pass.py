import pickle
import task6_technique
import pandas as pd
import numpy as np
from scipy.spatial import distance
import webbrowser,os,sys,time
import Task6_ppr
from argparse import ArgumentParser


def create_html(query_image,relevant,filename,img_folder):
    html_op = ("<html><head><title>%s</title></head><body><h2><b>Output for Task %d</b></h2>" % (filename,6))
    html_op += ("<h4>Relevant Images for Query Image:%s based on user feedback</h4>" % (query_image))
    html_op += ("<h4>Total Count: %d</h4>" % (len(relevant)))

    html_op += ("<table>")
    for i in range(len(relevant)):
        if i%6 == 0:
            if i != 0:
                html_op += ("</tr>")
            html_op += ("<tr>")
            html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s &nbsp; Score:%s</div></td>" % (img_folder, relevant[i][0],relevant[i][0][:-4],float(relevant[i][1])))
        else:
            html_op += ("<td><div style='text-align:center'><img src='%s%s' width=200 height=200>%s &nbsp; Score:%s</div></td>"%(img_folder, relevant[i][0],relevant[i][0][:-4],float(relevant[i][1])))

    html_op += ("</tr></table>")
    html_op += "</body></html>"

    file = open(filename,"w")
    file.write(html_op)
    webbrowser.open('file://' + os.path.realpath(filename))


def probabilistic(img_folder,output_file):
    with open('hog_11k_svd_query_image.pickle', 'rb') as f:
        query_image = pickle.load(f)

    with open('hog_11k_svd_similar_images.pickle', 'rb') as f:
        similar_image = pickle.load(f)

    with open( 'hog_11k_svd_20_relevant_images.pickle', 'rb') as f:
        top_20_image = pickle.load(f)

    final = []
    image_list= []
    for i in similar_image:
        image_list.append(i[0])
        row = []
        temp = []
        for j in i[1]:
            temp.append(j)
        row.extend(temp)
        final.append(row)
    similar = pd.DataFrame(final)

    mean_val = similar.mean(axis=0)

    final_result = []

    for i in range(len(final)):
        temp = []
        for j in final[i]:
            if j < mean_val.values[i]:
                temp.append(0)
            else:
                temp.append(1)
        final_result.append(temp)

    print("20 similar images for the query image %s" % (query_image[0]))
    for i in range(20):
        print(top_20_image[i][0], end=",")
    print("")
    user_feedback_relevant = input("Enter comma seperated list of relevant images: ")
    user_feedback_irrelevant = input("Enter comma seperated list of irrelevant images: ")

    feedback_relevant = user_feedback_relevant.split(",")
    feedback_irrelevant = user_feedback_irrelevant.split(",")

    relevant_list = []
    irrelevant_list = []
    for img in feedback_relevant:
        relevant_list.append(final_result[image_list.index(img)])
    for img in feedback_irrelevant:
        irrelevant_list.append(final_result[image_list.index(img)])

    minimal = 0.000001
    relevant_count = list()
    irrelevant_count = list()
    for i in range(len(relevant_list[0])):
        numerator = 0
        for j in range(len(relevant_list)):
            if relevant_list[j][i] == 1:
                numerator += 1
        relevant_count.append(numerator+minimal)

    for i in range(len(irrelevant_list[0])):
        denominator = 0
        for j in range(len(irrelevant_list)):
            if irrelevant_list[j][i] == 1:
                denominator += 1
        irrelevant_count.append(denominator+minimal)

    ratio = []
    for rel,irrel in zip(relevant_count,irrelevant_count):
        ratio.append(rel/irrel)

    result = []
    for i in range(len(final_result)):
        score = 0.0
        for j in range(len(final_result[0])):
            score += final_result[i][j]*ratio[j]
        result.append([image_list[i],score])
    relevant = sorted(result, key=lambda p: p[1],reverse=True)
    print("Relevant images based on user feedback:%d" % (len(relevant)))
    print(relevant)
    create_html(query_image[0], relevant, output_file, img_folder)

# Input arguments
def read_argument(argv):
    parser = ArgumentParser()
    parser.add_argument("-t", "--technique", type=str, help="Technique")
    parser.add_argument("-f", "--folder", type=str, help="Folder_path")
    return parser.parse_args(argv)

if __name__ == '__main__':
    start_time = time.time()
    args = read_argument(sys.argv[1:])
    img_folder = args.folder
    technique = args.technique
    # technique = "PPR"
    output_file = "task6_output.html"
    # img_folder = "/home/mansi/PycharmProjects/mwdb_phase3/Hands/"
    if technique == 'probabilistic':
        probabilistic(img_folder,output_file)

    else:

        with open( 'hog_11k_svd_query_image.pickle', 'rb') as f:
            query_image = pickle.load(f)

        with open( 'hog_11k_svd_similar_images.pickle', 'rb') as f:
            similar_image = pickle.load(f)

        task6_input = similar_image

        similar_image = np.array(similar_image)
        similar = pd.DataFrame(similar_image[:,1:], index=similar_image[:,0])

        with open( 'hog_11k_svd_20_relevant_images.pickle', 'rb') as f:
            top_20_image = pickle.load(f)

        print("20 similar images for the query image %s" % (query_image[0]))
        for i in range(20):
            print(top_20_image[i][0],end=",")
        print("")
        user_feedback_relevant = input("Enter comma seperated list of relevant images: ")
        user_feedback_irrelevant = input("Enter comma seperated list of irrelevant images: ")

        feedback_relevant = user_feedback_relevant.split(",")
        feedback_irrelevant = user_feedback_irrelevant.split(",")

        feedback_data_relevant = similar.loc[feedback_relevant]
        feedback_data_relevant["label"] = "relevant"
        feedback_data_irrelevant = similar.loc[feedback_irrelevant]
        feedback_data_irrelevant["label"] = "irrelevant"

        technique_op = list()
        feedback_data = pd.concat([feedback_data_relevant,feedback_data_irrelevant])
        if technique == "svm":
            x_train = feedback_data.iloc[:,0]
            y_train = feedback_data['label']
            x_test = similar.iloc[:, 0]
            technique_op = task6_technique.svm(x_train, y_train, x_test)

        elif technique == "decision_tree":
            train_data = list()
            test_data = list()
            for i in range(len(feedback_data)):
                train_row = list(feedback_data.iloc[i][0])
                train_row.append(feedback_data.iloc[i]["label"])
                train_data.append(train_row)

            for i in range(len(similar)):
                test_data.append(list(similar.iloc[i][0]))
            technique_op = task6_technique.train_test_data_dcsn(train_data, test_data)

        elif technique == "ppr":
            technique_op = Task6_ppr.task6_ppr(task6_input,feedback_relevant,feedback_irrelevant)
            # Task6_ppr.la(task6_input, feedback_relevant, feedback_irrelevant)

        relevant = list()
        for i in range(len(technique_op)):
            if technique_op[i] == 'relevant':
                img_score = list()
                img_score.append(similar_image[i][0])
                score = distance.euclidean(similar_image[i][1],query_image[1])
                img_score.append(score)
                relevant.append(img_score)
        relevant = sorted(relevant, key=lambda p: p[1])
        print("Relevant images based on user feedback:%d"%(len(relevant)))
        print(relevant)
        create_html(query_image[0],relevant,output_file,img_folder)
    print("Execution time:%s seconds"%(time.time()-start_time))
