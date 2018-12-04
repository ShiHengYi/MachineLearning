import email.parser
import os, sys, stat
import shutil
import numpy as np
import math
import scipy as sp
import re

class NB():
    def ExtractSubPayload(filename):
        if not os.path.exists(filename):  # dest path doesnot exist
            print("ERROR: input file does not exist:", filename)
            os.exit(1)
        fp = open(filename, encoding='iso-8859-1')
        payload =""
        for line in fp:
            payload += line
        stringre = re.sub('[^A-Za-z0-9 ]+', '', payload)
        return stringre

    def training(srcdir):
        #initialize variable
        spam_wordcount = 0
        ham_wordcount = 0
        spamDict = {}
        spam_mail_count = 0
        hamDict = {}
        ham_mail_count = 0
        #fileIO
        files = os.listdir(srcdir)
        label_file = open("SPAM.label")
        print ("Start to training ...")
        for file in files:
            srcpath = os.path.join(srcdir, file)
            label_line = label_file.readline()
            body = NB.ExtractSubPayload(srcpath)
            if(body != "-1"): #check if the text is valid, skip if it's not valid
                mailparser = body.split(" ")
                lb = label_line.split(" ")
                if (lb[0] == '1'): #check label for that e-mail
                    ham_mail_count = ham_mail_count + 1
                    for word in mailparser:
                        ham_wordcount += 1
                        if(word in hamDict):
                            hamDict[word] += 1
                        else:
                            hamDict[word] = 1
                else:
                    spam_mail_count = spam_mail_count + 1
                    for word in mailparser:
                        spam_wordcount += 1

                        if (word in spamDict):
                            spamDict[word] += 1
                        else:
                            spamDict[word] = 1
        #####################################################
        #this part use to purify the training data set      #
        keyset = []
        for key in spamDict:
            if (len(key) == 1):
                keyset.append(key)
            if (len(key) == 2):
                keyset.append(key)
            if (key == 'html'):
                keyset.append(key)
        for removekey in keyset:
            if removekey in spamDict:
                spam_wordcount -= spamDict[removekey]
                del spamDict[removekey]

        keyset = []
        for key in hamDict:
            if (len(key) == 1):
                keyset.append(key)
            if (len(key) == 2):
                keyset.append(key)
            if (key == 'html'):
                keyset.append(key)
        for removekey in keyset:
            if removekey in hamDict:
                ham_wordcount -= hamDict[removekey]
                del hamDict[removekey]
        #                                                   #
        #####################################################

        print("training finished")
        #pass variable for predicting phase
        NB.predicting(spam_wordcount, ham_wordcount, spamDict, spam_mail_count, hamDict, ham_mail_count)

    def predicting(a,b,c,d,e,f):
        print("Start to predicting ...")

        #initialize all variable
        spam_wordcount =a
        ham_wordcount = b
        spamDict = c
        spam_mail_count = d
        hamDict = e
        ham_mail_count = f
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        #file IO-------------------------------
        label_file = open("TEST.label")
        srcdir = "TESTING_PREP"
        files = os.listdir(srcdir)
        for file in files:
            pspam =0.0
            pham = 0.0
            decision = 0

            srcpath = os.path.join(srcdir, file)
            label_line = label_file.readline()
            body = NB.ExtractSubPayload(srcpath)
            mailparser = body.split(" ")
            lb = label_line.split(" ")

            for word in mailparser:
                if word in spamDict:
                    pspam += math.log10(spamDict[word]/spam_wordcount)

                if word in hamDict:
                    pham += math.log10(hamDict[word]/ham_wordcount)
            #probablity of ham vs spam
            ham_fra = ham_wordcount/(ham_wordcount+spam_wordcount)
            spam_fra = spam_wordcount/(ham_wordcount+spam_wordcount)
            #overall probablity score for spam = ps and ham = ph
            ps = pspam*spam_fra
            ph = pham*ham_fra

            #since both value are negative we compare the absolute value
            if (ps < ph):
                decision = 0
            else:
                decision = 1

            if(lb[0] == '1' and decision == 1):
                TP +=1
            elif(lb[0] == '1' and decision == 0):
                FN +=1
            elif(lb[0] == '0' and decision == 1):
                FP +=1
            else:
                TN +=1

        print("predicting finished")

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


    def main():
        NB.training("training_prep")

NB.main()

