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

def regex(string: str, model: str):
    """
    Function that computes regular expressions.
        All digits are set to 0s.
        All hyper links are simply replaced by the @ symbol
    """
    string = re.sub("[0-9]", "0", string)
    string = re.sub(r'( \n| \t)+', '', string)
    string = re.sub("https:\\\+([a-zA-Z0-9.]+)?", "@", string)
    
    if model == 'word':
        # if model is a word n-gram model, remove all punctuation
        string = ''.join([char for char in string if char.isalnum()])
        
    if model == 'char-dist':
        string = re.sub("[a-zA-Z]+", "*", string)
        # string = ''.join(['*' if char.isalpha() else char for char in string])
        
    return string

def frequency(tokens: list):
    """
    Count tokens in text (keys are tokens, values are their corresponding frequencies).
    """
    freq = dict()
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1
    return freq

def represent_text(text, n: int, model: str):
    """
    Extracts all character or word 'n'-grams from a given 'text'.
    Any digit is represented through a 0.
    Each hyperlink is replaced by an @ sign.
    The latter steps are computed through regular expressions.
    """ 
    if model == 'char-std':

        text = regex(text, model)
        tokens = [text[i:i+n] for i in range(len(text)-n+1)] 

        if n == 2:
            # create list of unigrams that only consists of punctuation marks
            # and extend tokens by that list
            punct_unigrams = [token for token in text if not token.isalnum()]
            tokens.extend(punct_unigrams)

    elif model == 'word':
        text = [regex(word, model) for word in text.split() if regex(word, model)]
        tokens = [' '.join(text[i:i+n]) for i in range(len(text)-n+1)]

    else:
        text = regex(text, model)
        tokens = tokens = [text[i:i+n] for i in range(len(text)-n+1)]
    
    freq = frequency(tokens)

    return freq

def extract_vocabulary(texts: list, n: int, ft: int, model: str):
    """
    Extracts all character 'n'-grams occurring at least 'ft' times in a set of 'texts'.
    """
    occurrences = {}
    
    for text in texts:

        text_occurrences=represent_text(text, n, model)
        
        for ngram in text_occurrences.keys():
            
            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]
    
    vocabulary=[]
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)
            
    return vocabulary

def extend_vocabulary(n_tuple: tuple, texts: list, model: str):
    n_start, n_range = n_tuple
    
    vocab = []
    for n in range(n_start, n_range + 1):
        n_vocab = extract_vocabulary(texts, n, (n_range - n) + 1, model)
        vocab.extend(n_vocab)
    return vocab

def pipeline(path,word_range: tuple, dist_range: tuple, char_range: tuple, n_best_factor = 0.3, pt=0.1, lower=False):
    print('Word n-gram range: ', word_range)
    print('Dist n-gram range: ', dist_range)
    print('Char n_gram raneg: ', char_range)
    
   
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
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
                
        # Building the training and test sets
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
            
        train_texts = [text for (text,label) in train_docs]        
        train_labels = [label for (text,label) in train_docs]
        test_docs = read_files(path+os.sep+problem,unk_folder)
        test_texts = [text for (text,label) in test_docs]
        
                
        print(problem)
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', len(test_texts), 'unknown texts')
        
        # Generating Vocabulary    
            # word n-gram vocabulary (content / semantical features)
        vocab_word = extend_vocabulary(word_range, train_texts, model = 'word')
        
            # character n-gram vocabulary (non-diacrictics / alphabetical symbols are distorted)
        vocab_char_dist = extend_vocabulary(dist_range, train_texts, model = 'char-dist')
           
            # character n-gram vocabulary (syntactical features)
        vocab_char_std = extend_vocabulary(char_range, train_texts, model = 'char-std')
            
        print('\t', 'word-based vocabulary size: ', len(vocab_word))
        print('\t', 'standard character-based vocabulary size: ', len(vocab_char_std))
        print('\t', 'non-alphabetical character vocabulary size: ', len(vocab_char_dist))

        
        ## Word N-gram model (captures content)
        vectorizer_word = TfidfVectorizer(analyzer = 'word', ngram_range = word_range, use_idf = True, 
                                          norm = 'l2', lowercase = lower, vocabulary = vocab_word, 
                                          smooth_idf = True, sublinear_tf = True)
        
        train_data_word = vectorizer_word.fit_transform(train_texts).toarray()
        n_best = int(len(vectorizer_word.idf_) * n_best_factor)
        idx_w = np.argsort(vectorizer_word.idf_)[:n_best]
        train_data_word = train_data_word[:, idx_w]

        test_data_word = vectorizer_word.transform(test_texts).toarray()
        test_data_word = test_data_word[:, idx_w]
        
        
        ## non-diacritics n-gram model (captures punctuation and meta-characters)
        vectorizer_char_dist = TfidfVectorizer(analyzer = 'char', ngram_range = dist_range, use_idf = True, 
                                     norm = 'l2', lowercase = lower, vocabulary = vocab_char_dist, 
                                     min_df = 0.2, max_df = 0.8, smooth_idf = True, 
                                     sublinear_tf = True)

        train_data_char_dist = vectorizer_char_dist.fit_transform(train_texts).toarray()
        n_best = int(len(vectorizer_char_dist.idf_) * n_best_factor)
        idx_c = np.argsort(vectorizer_char_dist.idf_)[:n_best]
        train_data_char_dist = train_data_char_dist[:, idx_c]

        test_data_char_dist = vectorizer_char_dist.transform(test_texts).toarray()
        test_data_char_dist = test_data_char_dist[:, idx_c]
        
        
        
        ##  Char n-gram model (captures syntactical features)
        vectorizer_char_std = TfidfVectorizer(analyzer = 'char', ngram_range = char_range, use_idf = True, 
                                     norm = 'l2', lowercase = lower, vocabulary = vocab_char_std, 
                                     min_df = 0.2, max_df = 0.8, smooth_idf = True, 
                                     sublinear_tf = True)

        train_data_char_std = vectorizer_char_std.fit_transform(train_texts).toarray()
        n_best = int(len(vectorizer_char_std.idf_) * n_best_factor)
        idx_c = np.argsort(vectorizer_char_std.idf_)[:n_best]
        train_data_char_std = train_data_char_std[:, idx_c]

        test_data_char_std = vectorizer_char_std.transform(test_texts).toarray()
        test_data_char_std = test_data_char_std[:, idx_c]
        
        
        ## Preprocessing
        
        # Data scaling 
        max_abs_scaler = preprocessing.MaxAbsScaler()
            # word n-gram model 
        scaled_train_data_word = max_abs_scaler.fit_transform(train_data_word)
        scaled_test_data_word = max_abs_scaler.transform(test_data_word)
        
            # char dist n-gram model 
        scaled_train_data_dist = max_abs_scaler.fit_transform(train_data_char_dist)
        scaled_test_data_dist = max_abs_scaler.transform(test_data_char_dist)
        
            # char std n-gram model 
        scaled_train_data_char = max_abs_scaler.fit_transform(train_data_char_std)
        scaled_test_data_char = max_abs_scaler.transform(test_data_char_std)
        
        
        # LSA 
        
        #svd = TruncatedSVD(n_components = 63, algorithm = 'randomized', random_state = 42)
        #scaled_train_data = svd.fit_transform(scaled_train_data)
        #scaled_test_data = svd.transform(scaled_test_data)
                

        ## Classification
        
        word = CalibratedClassifierCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 300),cv=5)
        word.fit(scaled_train_data_word, train_labels)
        preds_word = word.predict(scaled_test_data_word)
        probs_word = word.predict_proba(scaled_test_data_word)
        
        
        dist = CalibratedClassifierCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 300),cv=5)
        dist.fit(scaled_train_data_dist, train_labels)
        preds_dist = dist.predict(scaled_test_data_dist)
        probs_dist = dist.predict_proba(scaled_test_data_dist)
        
        
        char = CalibratedClassifierCV(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter = 300),cv=5)
        char.fit(scaled_train_data_char, train_labels)
        preds_char = char.predict(scaled_test_data_char)
        probs_char = char.predict_proba(scaled_test_data_char)
                 
    
        
        
        avg_probs = np.average([probs_word, probs_dist, probs_char], axis = 0)        
        avg_predictions = []
        for text_probs in avg_probs:
            ind_best = np.argmax(text_probs)
            avg_predictions.append(candidates[ind_best])
            
                  
                
        ensemble = LogisticRegression(random_state=0, solver='lbfgs', multi_class = 'multinomial', max_iter = 300)
        ensemble_train = np.concatenate((
                word.predict_proba(scaled_train_data_word), 
                dist.predict_proba(scaled_train_data_dist),
                char.predict_proba(scaled_train_data_char)            
            ), axis = 1)        
        ensemble_test = np.concatenate((
                probs_word, 
                probs_dist,
                probs_char
            ), axis = 1)       
        ensemble.fit(ensemble_train, train_labels)
        ensemble_predictions = ensemble.predict(ensemble_test)
        ensemble_proba  = ensemble.predict_proba(ensemble_test)
        
        
        
        # Reject option (used in open-set cases)
        for i, p in enumerate(preds_word):
            sproba=sorted(probs_word[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                preds_word[i]=u'<UNK>'
                
        for i, p in enumerate(preds_dist):
            sproba=sorted(probs_dist[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                preds_dist[i]=u'<UNK>'
                
        for i, p in enumerate(preds_char):
            sproba=sorted(probs_char[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                preds_char[i]=u'<UNK>'
                        
        for i,p in enumerate(avg_predictions):
            sproba=sorted(avg_probs[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                avg_predictions[i]=u'<UNK>'
       
        count=0
        for i,p in enumerate(ensemble_predictions):
            sproba=sorted(ensemble_proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                ensemble_predictions[i]=u'<UNK>'
                count=count+1
        print('\t',count,'texts left unattributed (by ensemble)')
        
        
        # Saving output data of word classfier
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(preds_word):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('word'+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        
        # Saving output data of dist classfier
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(preds_word):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('dist'+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        
        # Saving output data of char classfier
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(preds_word):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open('char'+os.sep+'answers-'+problem+'.json', 'w') as f:
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
        
        
    print()
    print('elapsed time:', time.time() - start_time)



def main():
    
    # pipeline(path,word_range: tuple, dist_range: tuple, char_range: tuple, n_best_factor = 0.3, pt=0.1, lower=False):
    
    word_range = (1,3)
    dist_range = (1,3)
    char_range = (2,5)
    
    
    pipeline(collection_folder,word_range, dist_range, char_range,n_best_factor = 0.4, pt=0.1, lower=False )
     
    print("\nResults for Word based classifier: ")
    evaluate_all(collection_folder,'word',evaluation_folder)

    print("\nResults for Dist based classifier: ")
    evaluate_all(collection_folder,'dist',evaluation_folder)

    print("\nResults for Char based classifier: ")
    evaluate_all(collection_folder,'char',evaluation_folder)

    print("\nResults for avg: ")
    evaluate_all(collection_folder,output_folder,evaluation_folder)
    
    print("\nResults for Ensemble: ")
    evaluate_all(collection_folder,'ensemble',evaluation_folder)


if __name__ == '__main__':
    main()