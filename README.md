# Authorship Attribution in Fan-Fictional Texts given variable length Character and Word N-Grams

If you would like to use parts of the code or replicate our approach, please cite us! :) 

Authors: Lukas Muttenthaler, Gordon Lucas, Janek Amann.

Authorship  Attribution  (AA)  is  the  task  of determining the author of a text from a set of candidates.  It requires text features to be represented  according  to  rigorous  experiments. In  the  context  of  machine  learning,  AA  can be   regarded   as   a   multi-class,   single-label text  classification  problem.    Its  applications include plagiarism detection and   forensic linguistics as well as research in literature.  

In the current study,  we aimed to develop three different n-gram models to identify authors of various fan-fictional texts.   Each of the three models  was  developed  as  a  variable-length n-gram model.  We implemented both a standard  character n-gram  model  (2−5 gram), 
a  distorted  character n-gram  model  (1−3 gram) and a word n-gram model (1−3 gram) to  not  only  capture  the  syntactic  features, but also the lexical features and content of a given  text.   Token  weighting  was  performed through term-frequency inverse-document frequency  (tf-idf)  computation.   For  each  of the  three  models,  we  implemented  a  linear Support   Vector   Machine   (SVM)   classifier, and in the end applied a soft voting procedure to  take  the  average  of  the  classifiers’  results (i.e.,  ensemble  SVM).  Results  showed,  that among   the   three   individual   models, the standard  character n-gram  model  performed best.   However, the  combination  of  all  three classifier’s predictions yielded the best results overall.  To enhance computational efficiency, we computed dimensionality reduction using Singular Value  Decomposition (SVD) before fitting  the  SVMs  with  training  data. 

With a  run  time  of  approximately 180 seconds for all 20 problems,  we achieved   a   macro   F1-score   of 70.5% for the  development  corpus  and  a  F1-score  of 69% for the competition’s test corpus, which significantly   outperformed   the   PAN   2019 baseline classifier.  Thus, we have shown that it is not a single feature representation that will yield  accurate  classifications,  but  rather  the combination  of  various  text  representations that will depict an author’s writing style most thoroughly.

The code for our approach can be found in the provided Jupyter notebook.
