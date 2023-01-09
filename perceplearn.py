import re   # is a standard python library    Ref: https://docs.python.org/3/library/
import numpy as np
import random
from random import shuffle
import sys


file=sys.argv[1]

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,.;]')
BAD_SYMBOLS_RE = re.compile('[^a-zA-Z #+_]')

def review_cleaning(text):
    text = text.lower() # converting all reviews to lowercase
    text = text.replace(r's*https?://S+(s+|$)', ' ').strip()
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text


def vanilla_perceptron(X,max_iterations,flag):
        weights={}
        for val in X[0]['feature_vector']:
            weights[val]=0
        bias=0
        final_bias=0
        final_weights={}
        tr_lst=[]
        highest_f1=0
        
        for iterations in range(max_iterations):
                shuffle(X)
                X_train=X
                for i in range(len(X_train)):
                    X_tr=X_train[i]['feature_vector']
                    if flag==0:
                        y=X_train[i]['label1']
                    else:
                        y=X_train[i]['label2']
                    a=sum([weights[val]*X_tr[val] for val in X_tr]) + bias
                    if y * a <=0:
                        for val in X_tr:
                                weights[val]+=(y * X_tr[val] )
                        bias+=y
                
                
        return weights,bias



def averaged_perceptron(X,max_iterations,flag):
        weights={}
        u={}
        final_weights={}
        final_weights2={}
        final_bias=0
        final_bias2=0
        for val in X[0]['feature_vector']:
            weights[val]=0
            u[val]=0
        bias=0
        beta=0
        tr_lst=[]
        highest_acc=0
        highest_f1=0
        c=1
        f1=0
        count=0
        for iterations in range(max_iterations):
                shuffle(X)
                X_train=X
                for i in range(len(X_train)):
                    X_tr=X_train[i]['feature_vector']
                    if flag==0:
                        y=X_train[i]['label1']
                    else:
                        y=X_train[i]['label2']
                    a=sum([weights[val]*X_tr[val] for val in X_tr]) + bias
                    if y * a <=0:
                        for val in X_tr:
                                weights[val]+=(y * X_tr[val])
                                u[val]+=(y*c*X_tr[val])
                        bias+=y
                        beta+=(y*c)
                    c+=1
                
                     
        for val in weights:
            final_weights[val]= weights[val]- ((1/c)*u[val])
        
        final_bias = bias - ((1/c)*beta)
        return final_weights,final_bias



def vanilla_test(X_test,weights,bias):
    pred=[]
    for i in range(len(X_test)):
        a=sum([weights[val] * X_test[i]['feature_vector'][val] for val in X_test[i]['feature_vector']]) + bias
        if a<=0:
            pred.append(-1)
        else:
            pred.append(1)
    return pred


with open(file,encoding='utf-8') as f:
    data = f.read().splitlines()

review_lst=[]
label1=[]
label2=[]

training_dict={}

for i in range(len(data)):
    key_id=data[i].split(' ')[0]
    training_dict[key_id]={}
    cleaned_text=review_cleaning(' '.join(data[i].split(' ')[3:]))
    cleaned_text=re.sub(' +', ' ', cleaned_text).strip()
    training_dict[key_id]['review'] =  cleaned_text
    if data[i].split(' ')[1]=="Fake":
        training_dict[key_id]['label1']= -1
    else:
        training_dict[key_id]['label1']=  1

    if data[i].split(' ')[2]=="Neg":
        training_dict[key_id]['label2']= -1
    else:
        training_dict[key_id]['label2']=  1



vocab={}
stop_word=set()
stop_word=set(['a','able','hotel','room','stay','chicago','city','michigan','about','she','with','this','you','our','rooms','across','all','also','am','among','an','and','any','are','as','at','be','because','been','by','dear','else','for','from','had','has','her','hers','him','his','i','in','into','is','it','me','my','of','on','only','or','own','so','that','they','the','their','there','to','us','was','we', 'were' ,'t' ,'s' ,'m', 'r' ,'ll', 've','he','she', 'nd' ,'e', 'el' ,'n', 'when' , 'where' ,'thechicagocritic', 'downtown', 'looking', 'them', 'candle','las', 'vegas', 'don'])

#creat stop words separately for fake and pos neg classes

vocab={}
for uniqid in training_dict:
    review=training_dict[uniqid]['review']
    word_lst=review.split(' ')
    for word in word_lst:
        if word not in stop_word and len(word)>1:
            if word not in vocab:
                vocab[word]=set()
                vocab[word].add(uniqid)
            else:
                vocab[word].add(uniqid)


tf={}
for uniqid in training_dict:
    review=training_dict[uniqid]['review']
    word_lst=review.split(' ')
    for word in vocab:
        tf[word,uniqid]=word_lst.count(word)/len(word_lst)
    
idf={}
num_documents=len(training_dict)

for word in vocab:
    idf[word]=np.log((1+num_documents)/(1+len(vocab[word]))) + 1



X_train=[]
for uniqid in training_dict:
     tr_dic={}
     vector={}
     scale_sum=0
     for word in vocab:
        vector[word]=tf[word,uniqid] * idf[word]
        scale_sum+= (tf[word,uniqid] * idf[word])**2
     for val in vector:
        vector[val]=vector[val]/ (scale_sum)**0.5
    
     tr_dic['feature_vector']=vector
     tr_dic['label1']=training_dict[uniqid]['label1']
     tr_dic['label2']=training_dict[uniqid]['label2']
     X_train.append(tr_dic)




weights_fake_true,bias_fake_true=vanilla_perceptron(X_train,35,0)   # 30 max
weights_pos_neg,bias_pos_neg=vanilla_perceptron(X_train,35,1)    # 25 max

avg_weights_fake_true,avg_bias_fake_true=averaged_perceptron(X_train,40,0)   # 30 max
avg_weights_pos_neg,avg_bias_pos_neg=averaged_perceptron(X_train,40,1)    # 35 max


model_dic={}
model_dic['vocab']=vocab 
model_dic['weights_fake_true']=weights_fake_true
model_dic['bias_fake_true']=bias_fake_true
model_dic['weights_pos_neg']=weights_pos_neg
model_dic['bias_pos_neg']=bias_pos_neg
model_dic['idf']=idf

avg_model_dic={}
avg_model_dic['vocab']=vocab
avg_model_dic['avg_weights_fake_true']=avg_weights_fake_true
avg_model_dic['avg_bias_fake_true']=avg_bias_fake_true
avg_model_dic['avg_weights_pos_neg']=avg_weights_pos_neg
avg_model_dic['avg_bias_pos_neg']=avg_bias_pos_neg
avg_model_dic['idf']=idf

with open('vanillamodel.txt', 'w') as file:
     file.write(str(model_dic))
        
with open('averagedmodel.txt', 'w') as file:
     file.write(str(avg_model_dic))



