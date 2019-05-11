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
from sklearn.linear_model import LogisticRegression

from pan19_cdaa_evaluator import evaluate_all



collection_folder = 'cross-domain-authorship-attribution-train'
output_folder = 'answers_gordon'
evaluation_folder = 'evaluation'



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

def pipeline(path,n_start = 2, n_range=3,n_best_factor = 0.3, pt=0.1, lowercase=False):
    print('n-gram range %d - %d' % (n_start,n_range))
    #print('word-gram range %d - %d' %(word_gram_start, word_gram_range))
    
    
    
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
        print(language[index])
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
        for n in range(n_start, n_range + 1):
            vocabulary = extract_vocabulary(train_texts,n, (n_range-n)+1)
            vocab.extend(vocabulary)

        
            # Creates the BoW weighted by TF-IDF    #max_df = 0.9, min_df = 0.1
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(n_start,n_range), use_idf = True, sublinear_tf = True, lowercase=lowercase,vocabulary=vocab, norm='l2')
        train_data = vectorizer.fit_transform(train_texts)
        
        
        
        n_best = int(len(vectorizer.idf_) * n_best_factor)
        indexes = np.argsort(vectorizer.idf_)[:n_best]
        train_data = train_data[:,indexes]
        
            
            # Prints out statistics about each training text
        
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocab))
        
            # Building test set
        test_docs = read_files(path+os.sep+problem,unk_folder)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]
        
        test_data = vectorizer.transform(test_texts)
        test_data = test_data[:,indexes]

            # Prints out the nr of test texts
        print('\t', len(test_texts), 'unknown texts')
        
        
        # preprocessing 
                
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        
        
        # LSA 
        
        #svd = TruncatedSVD(n_components = 63, algorithm = 'randomized', random_state = 42)
        #scaled_train_data = svd.fit_transform(scaled_train_data)
        #scaled_test_data = svd.transform(scaled_test_data)
                

        # Applying SVM     # Maybe change to ensemble logistic classifier
        
        lgr = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 300)
        svc = OneVsRestClassifier(SVC(C=1, gamma='auto'))

      
        clf=CalibratedClassifierCV(lgr, cv = 5)  
        clf.fit(scaled_train_data, train_labels)
        predictions=clf.predict(scaled_test_data)
        proba=clf.predict_proba(scaled_test_data)

        
        clf2 = CalibratedClassifierCV(svc, cv = 5)
        clf2.fit(scaled_train_data, train_labels)
        predictions2 = clf2.predict(scaled_test_data)
        proba2 = clf2.predict_proba(scaled_test_data)
        
        
        
        avg_proba = np.average([proba, proba2], axis = 0)        
        avg_predictions = []
        for text_probas in avg_proba:
            ind_best = np.argmax(text_probas)
            avg_predictions.append(candidates[ind_best])
            
                  
        ensemble = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial', max_iter = 300)
        
        ensemble_train = np.concatenate(
            (clf.predict_proba(scaled_train_data), clf2.predict_proba(scaled_train_data)), axis = 1)        
        ensemble_test = np.concatenate((proba, proba2), axis = 1)       
        ensemble.fit(ensemble_train, train_labels)
        ensemble_predictions = ensemble.predict(ensemble_test)
        ensemble_proba  = ensemble.predict_proba(ensemble_test)
        
        
        #print(predictions[:5])
        #print(predictions2[:5])
        #print(avg_predictions[:5])
        #print(ensemble_predictions[:5])
        
        
        
        
        # Reject option (used in open-set cases)
        count=0
        
        for i,p in enumerate(predictions):
            sproba=sorted(avg_proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                avg_predictions[i]=u'<UNK>'
                
        for i,p in enumerate(predictions2):
            sproba=sorted(proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                avg_predictions[i]=u'<UNK>'
                
        for i,p in enumerate(avg_predictions):
            sproba=sorted(proba2[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                avg_predictions[i]=u'<UNK>'
        
        for i,p in enumerate(ensemble_predictions):
            sproba=sorted(ensemble_proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                avg_predictions[i]=u'<UNK>'
                count=count+1
        
        print('\t',count,'texts left unattributed (by ensemble)')
        
        
        # Saving output data of clf 1
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('classifier1'+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
            
        # Saving output data of clf 2
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions2):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('classifier2'+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
            
        # Saving output data of avg
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(avg_predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(output_folder+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)    
            
        # Saving output data of ensemble
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(ensemble_predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('ensemble'+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        
        
    
    
    print('elapsed time:', time.time() - start_time)



def main():
    pipeline(collection_folder,n_start = 2, n_range = 5,n_best_factor = 0.25, pt=0.1, lowercase=False )
     
    print("\nResults for classifier 1 (LGR): ")
    evaluate_all(collection_folder,'classifier1',evaluation_folder)

    print("\nResults for classifier 2 (SVM): ")
    evaluate_all(collection_folder,'classifier2',evaluation_folder)

    print("\nResults for avg: ")
    evaluate_all(collection_folder,output_folder,evaluation_folder)
    
    print("\nResults for Ensemble: ")
    evaluate_all(collection_folder,'ensemble',evaluation_folder)


if __name__ == '__main__':
    main()