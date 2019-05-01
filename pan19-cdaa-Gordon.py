# -*- coding: utf-8 -*-

"""
 Usage from command line: 
    > python pan19-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-pt PROBABILITY-THRESHOLD]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-19 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-19 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the cutoff threshold used to filter out rare n-grams (default=5)
   PROBABILITY-THRESHOLD (float) is the threshold for the reject option assigning test documents to the <UNK> class (default=0.1)
                                 Let P1 and P2 be the two maximum probabilities of training classes for a test document. If P1-P2<pt then the test document is assigned to the <UNK> class.
   
 Example:
     > python pan19-cdaa-baseline.py -i "mydata/pan19-cdaa-development-corpus" -o "mydata/pan19-answers"
"""

from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
import re
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, TruncatedSVD

def represent_text(text,n):
    """
    Extracts all character 'n'-grams from a given 'text'.
    Each digit is represented as a hashtag symbol (#) which in general denotes any number.
    Each hyperlink is replaced by an @ sign.
    The latter steps are computed through regular expressions.
    """    
    if n > 0:
        text = re.sub("[0-9]+(([.,^])[0-9]+)?", "#", text)
        text = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", text)
        tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    
    # create frequency text representation (keys are tokens, values are their corresponding frequencies)
    frequency = {token: tokens.count(token) for token in list(set(tokens))}
        
    return frequency

def read_files(path,label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(path+os.sep+label+os.sep+'*.txt')
    texts=[]
    for i,v in enumerate(files):
        f=codecs.open(v,'r',encoding='utf-8')
        texts.append((f.read(),label))
        f.close()
    return texts

def extract_vocabulary(texts,n,ft=3):
    # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
    occurrences=defaultdict(int)
    for text in texts:
        text_occurrences=represent_text(text,n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram]+=text_occurrences[ngram]
            else:
                occurrences[ngram]=text_occurrences[ngram]
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i]>=ft:
            vocabulary.append(i)
    return vocabulary

def pipeline(path,outpath,n_range=3,pt=0.1, lowercase=False):
    start_time = time.time()
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    for index,problem in enumerate(problems):
        print(problem)
        
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        
        # Building training set
        train_docs=[]
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
        train_texts = [text for i,(text,label) in enumerate(train_docs)]
        train_labels = [label for i,(text,label) in enumerate(train_docs)]
        
        # Builds the vocabulary for all n-Grams in between [2, n_range]
        # It discards n-Grams that do not occur frequently enough 
        vocab = []
        for n in range(2, n_range + 1):
            vocabulary = extract_vocabulary(train_texts,n, (n_range-n)+1)
            vocab.extend(vocabulary)

            # Creates the BoW weighted by TF-IDF    #max_df = 0.9, min_df = 0.1
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,n_range),use_idf = True, lowercase=lowercase,vocabulary=vocab, norm='l2')
        train_data = vectorizer.fit_transform(train_texts)
        
        
        n_best = int(len(vectorizer.idf_) * 0.25)
        indexes = np.argsort(vectorizer.idf_)[:n_best]
        train_data = train_data[:,indexes]
        
            # Reduce the dimensionality of the data to x % of the training data
            #n_components= int(len(vocab) * 0.1)
        #SVD = TruncatedSVD(50, random_state = 42)
        #train_data = SVD.fit_transform(train_data)

            # Prints out statistics about each training text
        
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocab))
        
        # Building test set
        test_docs = read_files(path+os.sep+problem,unk_folder)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]
        
        test_data = vectorizer.transform(test_texts)
        #test_data = SVD.transform(test_data)
        
        # Prints out the nr of test texts
        print('\t', len(test_texts), 'unknown texts')
        
        test_data = test_data[:,indexes]
        
        # preprocessing 
        
        
        # Some preprocessing that we dont need
        #for i,v in enumerate(train_texts):
        #   train_data[i]=train_data[i]/len(train_texts[i])
        #for i,v in enumerate(test_texts):
        #    test_data[i]=test_data[i]/len(test_texts[i])
        
        
        
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)

        # Applying SVM     # Maybe change to ensemble logistic classifier
        
        clf=CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1, gamma='auto')))
        clf.fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        proba=clf.predict_proba(scaled_test_data)
        
        # Reject option (used in open-set cases)
        count=0
        for i,p in enumerate(predictions):
            sproba=sorted(proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                predictions[i]=u'<UNK>'
                count=count+1
        print('\t',count,'texts left unattributed')
        
        # Saving output data
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file','answers-'+problem+'.json')
    print('elapsed time:', time.time() - start_time)




collection_folder = 'cross-domain-authorship-attribution-train'
output_folder = 'answers_gordon'

def main():
    pipeline(collection_folder,output_folder, n_range = 5,pt=0.1 )



if __name__ == '__main__':
    main()