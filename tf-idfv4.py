# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:00:37 2018

@author: Hugo Xian
"""
import jieba
import pickle 
#from collections import Counter
from gensim import corpora,models,similarities
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from collections import  defaultdict 
f1=open('E:\Dissertation\TextP\puretext_xy2.pickle','rb')
dataset=pickle.load(f1)
f1.close

"""
content1=dataset['x'][1]                           
cut=jieba.cut(content1)
cutword=[x for x in cut]
highfreq=Counter(cutword).most_common()
#print("/".join(cutword))
"""
total_cutword_x=[[word for word in jieba.cut(dataset['case_x'][i])] for i in range(0,len(dataset['case_x']))]#
test_case_x=[word for word in jieba.cut(dataset['case_x'][5])]
"""
count_vectorizer=CountVectorizer(min_df=50)
word_matrix_x=count_vectorizer.fit_transform(total_cutword_x[:])
x_matrix=word_matrix_x.toarray() 
x_vo_dict=count_vectorizer.vocabulary_  
x_vo_list=count_vectorizer.get_feature_names()
"""
x_dictionary=corpora.Dictionary(total_cutword_x)
x_dictionary.filter_extremes(no_below=50);len(x_dictionary)
xxdic=[word for word in x_dictionary.token2id]
xxfre=[fre for fre in x_dictionary.dfs]
x_corpus=[x_dictionary.doc2bow(case) for case in total_cutword_x] ##将语料库表示为向量
"""
frequency=defaultdict(int)
for text in total_cutword_x:
    for token in text:
        frequency[token]+=1
texts=[ [ token for token in text if frequency[token] > 5 ] for text in total_cutword_x]
"""
test_case_vec=x_dictionary.doc2bow(test_case_x)
tf_idf=models.TfidfModel(x_corpus)
case_vector=tf_idf[x_corpus]

index = similarities.SparseMatrixSimilarity(tf_idf[x_corpus], num_features=len(x_dictionary.keys()))
sim = index[tf_idf[test_case_vec]]
similarity=sorted(enumerate(sim), key=lambda item: -item[1])#[0:20]
deletecase=[];
for s in range(0,len(similarity)):
    if similarity[s][1]<0.10:
        deletecase.append(similarity[s])
for x in deletecase:
    similarity.remove(x)
#similarity.remove(x for x in deletecase)
simi_text_x_raw=[];simi_text_y=[];
for i in range(0,len(similarity)):
    if youtcome[0][similarity[i][0]] != 0 and youtcome[0][similarity[i][0]] != '*' \
    and dataset['case_x'][similarity[i][0]] not in simi_text_x_raw:
        simi_text_x_raw.append(dataset['case_x'][similarity[i][0]]);
        simi_text_y.append(youtcome[0][similarity[i][0]]);
###分成1和-1两部分文本
simi_text_x_1raw=[];simi_text_x_m1raw=[];simi_text_y_1=[];simi_text_y_m1=[];
for i in range(0,len(simi_text_x_raw)):
    if simi_text_y[i]==1:
        simi_text_x_1raw.append(simi_text_x_raw[i]);simi_text_y_1.append(simi_text_y[i]);
    else :
        simi_text_x_m1raw.append(simi_text_x_raw[i]);simi_text_y_m1.append(simi_text_y[i]);
#将文本分成两部分
from sklearn.cross_validation import train_test_split
def split_data(simi_text_x_raw,simi_text_y):
    return (train_test_split(simi_text_x_raw,simi_text_y,test_size=0.20))
#提取x文本特征向量
from sklearn.feature_extraction.text import TfidfVectorizer
def text_vector(x_train_raw):
    vectorizer=TfidfVectorizer(token_pattern='\w', ngram_range=(1,2), max_df=100, min_df=0)
    x_train=vectorizer.fit_transform(x_train_raw)
    return (x_train)

##one-calss svm
from sklearn import svm;import pandas as pd
def one_class_svm(simi_text_x_1raw,simi_text_y_1):
    simi_text_x_1=text_vector(simi_text_x_1raw)
    clf_svm = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_svm.fit(simi_text_x_1);
    y1_predict_train=clf_svm.predict(simi_text_x_1);
    print('outliers:{0}; total size:{1}'.format(y1_predict_train[y1_predict_train == -1].size,len(simi_text_x_1raw)))
    """
    simi_text_x=[dataset['x'][5],dataset['x'][1815],dataset['x'][12972],dataset['x'][3456],dataset['x'][13248]]
    simi_text_y=[youtcome[0][5],youtcome[0][1815],youtcome[0][12972],youtcome[0][3456],youtcome[0][13248]]
    """
    #查看逻辑异判案件    
    x_dev_punctuation=pd.DataFrame(columns=['case_x_punctuation','case_x','case_y','name']);
    for x_dev_raw_i in simi_text_x_1raw :
        x_dev_punctuation=x_dev_punctuation.append(dataset.loc[dataset.case_x == x_dev_raw_i]);
    x_dev_punctuation=x_dev_punctuation.drop_duplicates(subset=['case_x'],keep='first');##初始文本加上
    ##构造dataframe查看总的结果
    data_dev=pd.DataFrame(columns=['case_x_punctuation','case_x','y_raw','y_dev','y_predicted'])
    data_dev['case_x_punctuation']=x_dev_punctuation['case_x_punctuation'].tolist()
    data_dev['y_raw']=x_dev_punctuation['case_y'].tolist();
    data_dev['case_x']=x_dev_punctuation['case_x'].tolist();
    data_dev['y_dev']=simi_text_y_1;data_dev['y_predicted']=y1_predict_train
    #data_dev.to_excel('E:/Dissertation/TextP/tf-idfv3.xlsx')
    return (data_dev)
data_dev_1=one_class_svm(simi_text_x_1raw,simi_text_y_1);
data_dev_m1=one_class_svm(simi_text_x_m1raw,simi_text_y_m1);
data_dev_1.to_excel('E:/Dissertation/TextP/tf-idfv4_1.xlsx');data_dev_m1.to_excel('E:/Dissertation/TextP/tf-idfv4_m1.xlsx')
