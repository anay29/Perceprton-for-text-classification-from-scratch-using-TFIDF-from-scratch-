import re   # is a standard python library    Ref: https://docs.python.org/3/library/
import numpy as np
import random
from random import shuffle
import sys

model_file=sys.argv[1]
data_file=sys.argv[2]

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')


def vanilla_test(X_test,weights,bias):
    pred=[]
    for i in range(len(X_test)):
        a=sum([weights[val] * X_test[i]['feature_vector'][val] for val in X_test[i]['feature_vector'] if val in weights]) + bias
        if a<=0:
            pred.append(-1)
        else:
            pred.append(1)
    return pred

def review_cleaning(text):
    text = text.lower() # converting all reviews to lowercase
    text = text.replace(r's*https?://S+(s+|$)', ' ').strip()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text


with open(data_file,encoding='utf-8') as f:
    test_data = f.read().splitlines()


with open(model_file,encoding='utf-8') as file:
    model_dic = file.read()


model_dic=eval(model_dic)
vocab=model_dic['vocab']

if model_file=="vanillamodel.txt":
    weights_fake_true=model_dic['weights_fake_true']
    bias_fake_true=model_dic['bias_fake_true']
    weights_pos_neg=model_dic['weights_pos_neg']
    bias_pos_neg=model_dic['bias_pos_neg']
    idf=model_dic['idf']
else:
    avg_weights_fake_true=model_dic['avg_weights_fake_true']
    avg_bias_fake_true=model_dic['avg_bias_fake_true']
    avg_weights_pos_neg=model_dic['avg_weights_pos_neg']
    avg_bias_pos_neg=model_dic['avg_bias_pos_neg']
    idf=model_dic['idf']


testing_dict={}

for i in range(len(test_data)):
    key_id=test_data[i].split(' ')[0]
    testing_dict[key_id]={}
    cleaned_text=review_cleaning(' '.join(test_data[i].split(' ')[1:]))
    cleaned_text=re.sub(' +', ' ', cleaned_text).strip()
    testing_dict[key_id]['review'] =  cleaned_text


vocab={}
for uniqid in testing_dict:
    review=testing_dict[uniqid]['review']
    word_lst=review.split(' ')
    for word in word_lst:
            if word not in vocab:
                vocab[word]=set()
                vocab[word].add(uniqid)
            else:
                vocab[word].add(uniqid)  
  
tf={}
for uniqid in testing_dict:
    review=testing_dict[uniqid]['review']
    word_lst=review.split(' ')
    for word in vocab:
        tf[word,uniqid]=word_lst.count(word)/len(word_lst)
        
X_test=[]
for uniqid in testing_dict:
     tr_dic={}
     vector={}
     scale_sum=0
     for word in vocab:
        if word not in idf:
            vector[word]=0
        else:
            vector[word]=tf[word,uniqid] * idf[word]
        scale_sum+= vector[word]**2
     for val in vector:
        vector[val]=vector[val]/ (scale_sum)**0.5
     tr_dic['feature_vector']=vector
     X_test.append(tr_dic)


if model_file=="vanillamodel.txt":
    predictions_fake_true=vanilla_test(X_test,weights_fake_true,bias_fake_true)
    predictions_pos_neg=vanilla_test(X_test,weights_pos_neg,bias_pos_neg)
    final_predictions_fake_true = ['Fake' if int(val)== -1 else 'True' for val in predictions_fake_true]
    final_predictions_pos_neg = ['Neg' if int(val)== -1 else 'Pos' for val in predictions_pos_neg]
    i=0
    lst=[]
    for uniqid in testing_dict:
        st=""
        st=uniqid+" "+str(final_predictions_fake_true[i])+" "+str(final_predictions_pos_neg[i])
        lst.append(st)
        i+=1
        
    with open("percepoutput.txt", 'w', encoding='utf-8') as f:
        for line in lst:
            f.write(f"{line}\n")

else:
    avg_predictions_fake_true=vanilla_test(X_test,avg_weights_fake_true,avg_bias_fake_true)
    avg_predictions_pos_neg=vanilla_test(X_test,avg_weights_pos_neg,avg_bias_pos_neg)
    final_avg_predictions_fake_true = ['Fake' if int(val)== -1 else 'True' for val in avg_predictions_fake_true]
    final_avg_predictions_pos_neg = ['Neg' if int(val)== -1 else 'Pos' for val in avg_predictions_pos_neg]
    
    i=0
    lst=[]
    for uniqid in testing_dict:
        st=""
        st=uniqid+" "+str(final_avg_predictions_fake_true[i])+" "+str(final_avg_predictions_pos_neg[i])
        lst.append(st)
        i+=1
        
    with open("percepoutput.txt", 'w', encoding='utf-8') as f:
        for line in lst:
            f.write(f"{line}\n")



