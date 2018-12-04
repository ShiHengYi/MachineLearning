import math
import os
import sys
import string
import collections
import numpy as np
from sklearn import svm

max_np_size = 3000 #mem error if size is too big

def make_word_dict(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    word_list = []
    for email in emails:
        with open(email, encoding='iso-8859-1') as lines:
            for line in lines:
                line_parser = [word.strip(string.punctuation) for word in line.split()]
                word_list += line_parser
    #sort word prequency so i can later on use the most valuable attribute first
    sorted_word_list = collections.Counter(word_list)

    #purify the data set by remove useless data from sorted word list
    for word in list(sorted_word_list):
        if word.isalpha() == False:
            del sorted_word_list[word]
        elif len(word) == 1:
            del sorted_word_list[word]
    # only use first "max size" of attribute
    sorted_word_list = sorted_word_list.most_common(max_np_size)

    word_dict = {}
    #build a sorted_word_list, looks like{0: apple, 1:banana} the word is within in decending frequency
    for i in range(0,len(sorted_word_list)):
        word_dict[sorted_word_list[i][0]] = i
    return word_dict

#create a label list [0,1,...,0] from spam_label
def create_label_list(label_file):
    f = open(label_file)
    label_list = []
    for line in f:
        label_list.append(line.split(" ")[0])
    return label_list


#vectorize the .eml data to NP since the SVM library specific use NP as input

def join_NP_input(file_dir, word_dict):

    emails = [os.path.join(file_dir, f) for f in os.listdir(file_dir)]
    joined_NP = []
    for email in emails:
        features = np.zeros((max_np_size,), dtype=int)
        e = open(email, encoding='iso-8859-1')
        for line in e:
            # use the similar logic as "make_word_dict" to purify data by remove useless data
            line_parser = [word.strip(string.punctuation) for word in line.split(" ")]
            for word in line_parser:
                if word.isalpha() and len(word) > 1:
                    dic_value = word_dict.get(word)
                    #turn the sign on if word within in dictionary
                    if dic_value is not None:
                        features[dic_value] = 1

        joined_NP.append(features)
    return joined_NP


def main():
    print("Preprocessing Start:")
    train_dir = "training_prep"
    test_dir = "testing_prep"
    cross_dir = "crossvalidation"
    spam_label = "spam.label"
    test_label = "test.label"

    dict = make_word_dict(train_dir)
    rowinput = join_NP_input(train_dir, dict)
    columninput = create_label_list(spam_label)
    print("Start training phase...")
    clf = svm.SVC()
    clf.fit(rowinput, columninput)
    print("Training Complete")

    '''
    print("Start cross-validation phase...")
    cross = join_NP_input(cross_dir,dict)
    validation = clf.predict(cross)
    print("Cross validation complete")
    '''

    print("Start predicting phase...")
    testing = join_NP_input(test_dir,dict)
    predictions = clf.predict(testing)
    print("Finishing prediction, providing statistical data: ")

    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    testing_list = create_label_list(test_label)

    for n in range(0,len(testing_list)):
        if (testing_list[n] == '1' and predictions[n] == '1'):
            TP += 1
        elif (testing_list[n] == '1' and predictions[n] == '0'):
            FN += 1
        elif (testing_list[n] == '0' and predictions[n] == '1'):
            FP += 1
        else:
            TN += 1
    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))
    prec = TP / (TP + FP)
    recall = TP / (TP + FN)
    print("False Positive Rate: " + str(FP / (FP + TN)))
    print("False Negative Rate: " + str(FN / (FN + TP)))
    print("Recall: " + str(recall))
    print("Precision: " + str(prec))
    print("F-Score: " + str((2 * prec * recall) / (prec + recall)))

    '''
    pre3_4 = join_NP_input("3_4",dict)
    instance3_4 = clf.predict(pre3_4)
    print(instance3_4[0])
    '''
main()
