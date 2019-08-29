# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:16:09 2019

@author: Ian
"""

import os
import numpy as np
from collections import counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernouliNB
from sklearn.svm import SVC, NuSVC, LinearSVC

#we are reading all data present in directory
def make_Dictionary(train_dir):
    emails = [os.path.join(tran_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m:
            for i, line in enumerate(m):
                if i == 2: #first row is header so we are starting at row 2
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)
    #remove non-words from data set
    list_to_remove = dictionary.keys()
    keys_to_remove = []
    
    for k in list_to_remove:
        if k.isalpha() == False:
            key_to_remove.append(k)
        elif len(k) == 1:
            keys_to_remove.append(k)
            
    for item in keys_to_remove:
        del dictionary[item]
    
    dictionary = dictionary.most_common(3000)
    #dictionarty containing 300 most common words in dataset
    return dictionary

#extract fatures of training data
def extract_features(mail_dir):
    files = [os.path.join(mail_dir, file_descriptor) for file_descriptor in os.listdir(mail_dir)
    features_matrix = np.zeros((len(files),3000))
    doc_id = 0
    for file_to_read in files:
        with open(file_to_read) as file_descriptor:
            for i, line in enumerate(file_descriptor):
                if i == 2:
                    words = line.split()
                    for word in words:
                        word_id = 0
                        for i, id in enumerate(dictionary):
                            if id[0] == word:
                                word_id = i
                                features_matrix[doc_id, word_id] = words.count(word)
            doc_id = doc_id + 1
    return features_matrix





train_dir = 'train-mail'
print("Process Training!")
dictionary = make_Dictionary(train_dir)

print("\nExtracting Features!")
train_labels = np.zeros(702)
train_labels[351:701] = 1
#get features from data
train_matrix = extract_features(train_dir)


#train SVM(support vector machine)(algorithm) and naive bayes classifier
print("\nTrain out system with SVM and Naive Bayes!")
multinomia_model = MultinomialNB()
svc_model = LinearSVC()
multinomia_model.fit(train_matrix, train_labels)
svc_model.fit(train_matrix.train_labels)

#trst unseen mail for spam
test_dir = 'test-mail'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)#number of emails in training and testmail folders
test_labels[130:260] = 1
result1 = multinomia_model.predict(test_matrix)
result2 = svc_model.predict(test_matrix)
print("Our results are as follows:")
print("0 represents no spam while 1 represents spam.")
print(result1)
print(result2)





