import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
import random
import time
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class DecisionTree():
    def __init__(self):
        self.tree = {}
    def learn(self, training_set,index,total_attributes,attributes):
        value=pos_a=pos_b=0
        random_feature_list= random.sample(total_attributes,attributes)
        gain,column_name,value = get_best_split(training_set,random_feature_list)
        if gain==0: return build_tree(training_set, column_name, value)
        corr_data, incor_data = divide_data(training_set,column_name,value)
        pos_a = self.learn(corr_data, 2*index,total_attributes,attributes)        
        pos_b= self.learn(corr_data,2*index+1,total_attributes,attributes)
        self.tree[index]=[column_name,value,{"corr_data":pos_a,"incor_data":pos_b}]
        return index
    
    def classify(self, test_instance):
        res = 0 # base
        index=1 # root 
        while 1:
            value= self.tree[index]
            num= value[0]
            values=value[1]
            if float(test_instance[num])>=float(values):
                res= value[2]["corr_data"]  
            else:
                res=value[2]["incor_data"]
            if res=="class1" or res=="class0":
                break
            else:
                index=res
        if res=="class1":
            res="1"
        else:
            res="0"
        return res

def get_best_split(training_data,total_attributes):
    temp_entropy=0
    top_entropy=1
    top_col=0
    best_avg=0
    top_ig=0
    entropy_before=entropy_temp(training_data)
    for j in total_attributes:
        data= [float(i[j]) for i in training_data]
        avg=float(np.sum(data)/len(data)) 
        corr_data, incor_data= divide_data(training_data,j,avg)
        if(len(corr_data) == 0 or len(incor_data)==0):
            continue
        entropy_after= curr_entropy(training_data,j,avg)
        temp_ig= cal_info_gain(entropy_before,entropy_after)
        if temp_ig>= top_ig:
            top_ig=temp_ig
            top_entropy=temp_entropy
            top_col = j
            best_avg=avg
    return  top_ig,top_col,best_avg

def divide_data(training_data, j, ans):
    corr_data= [i for i in training_data if float(i[j]) >= ans]
    incor_data = [i for i in training_data if float(i[j])< ans]
    return corr_data, incor_data

def build_tree(training_data, j,ans):
    flag_1= None
    flag_0=None
    corr_data= [i for i in training_data if float(i[j]) >= ans]
    incor_data = [i for i in training_data if float(i[j])< ans]
    if len(corr_data)>0:
        flag_1 = corr_data[0][-1]
    if len(incor_data)>0:
        flag_0= incor_data[0][-1]     
    if flag_0==1:
        flag_0="class1"
    else:
        flag_0="class0"
    if flag_1==0:
        flag_1="class0"
    else:
        flag_1="class1"
    if len(corr_data)> len(incor_data):
        return flag_1
    else:
        return flag_0

def cal_entropy(training_data):
    if(training_data.shape[0]==0):
        return 0
    Y_train=training_data[:,-1]
    unique_elements, counts_elements = np.unique(Y_train, return_counts=True) 
    dic=dict(zip(unique_elements,counts_elements)) 
    p = []
    for label in dic:
        p.append(dic[label]/float(Y_train.shape[0])) 
    return np.sum(np.negative(p) * np.log2(p))

def entropy_temp(training_data):
    training_data=np.asarray(training_data).astype("float")
    temp_entropy = cal_entropy(training_data)
    return temp_entropy

def curr_entropy(training_data, j, ans):
    left_child = [i for i in training_data if float(i[j]) >= ans]
    right_child = [i for i in training_data if float(i[j])< ans]
    left_child = np.asarray(left_child,dtype='float')
    right_child = np.asarray(right_child,dtype='float')
    p =float((left_child.shape[0])/(left_child.shape[0]+right_child.shape[0]))
    return np.negative(p*np.log2(p)+(1-p)*np.log2(1-p))

def cal_info_gain(entropy_before, entropy_after):
    return (entropy_before-entropy_after)

def data_simplification(whole_train,total_samples):
    whole_train= whole_train.sample(n=total_samples)
    selected_train = whole_train.iloc[:, :-1]
    total_attributes=[m for m in range(selected_train.shape[1])]
    whole_train=whole_train.values
    return whole_train,total_attributes

def build_RF(whole_train, attributes, total_cnt, total_samples):
    rf=[]
    for i in range(total_cnt):
        given_data,total_attributes= data_simplification(whole_train,total_samples)
        data=given_data.tolist()
        tree = DecisionTree()
        tree.learn(data,1,total_attributes,attributes)
        rf.append(tree)
    return rf

def cal_accuracy(testing_data, rf):
    strings=[]
    testing_data=testing_data.values.tolist()
    for example in testing_data:
        temp_ans=[]
        for tree in rf:
            ans=tree.classify(example)
            temp_ans.append(ans)
        strings.append(temp_ans)
    final_ans=[]
    for i in range(len(strings)):
        final_ans.append(max(set(strings[i]), key=strings[i].count))
    
    cnt=0   
    for i in range(len(final_ans)):
        if float(final_ans[i])==(testing_data[i][-1]):
            cnt=cnt+1
            accuracy= (cnt)/(len(final_ans))
    print("Accuracy of Random Forest calculated from scratch is : " + str(round((accuracy),5)))
    return accuracy

def accuracy_using_scikitlearn(ran,count_trees,training_data,testing_data):
    clf = RandomForestClassifier(n_estimators=count_trees,max_features='sqrt')
    Y_train=training_data.iloc[:,-1]
    X_train=training_data.iloc[:,:-1]
    Y_test=testing_data.iloc[:,-1]
    X_test=testing_data.iloc[:,:-1]
    clf.fit(X_train,Y_train)
    predict= clf.predict(X_test)
    accuracy = accuracy_score(Y_test,predict)
    print("Accuracy of Random Forest using Scikit-learn library is : " + str(round((accuracy),5)))
    return accuracy

#Data paths are given as per file locations in PC
def Random_Forest():
    load= pd.read_csv('D:\spam.data',header=None,delimiter=r"\s+")
    training_data, testing_data = train_test_split(load, test_size=0.3)
    M = training_data.shape[1]
    features=M
    m=int(math.sqrt(M))
    reduced_features = m
    bootstrap=1534
    tree_list=[]
    scratch_accuracy_list=[]
    scikitlearn_accuracy_list=[]
    rand_time=0
    scratch_algo_time=0
    scikitlearn_time=0
    ans=0
    total_trees = [5,8,13,22,37,45,50,67,80,100]
    print("**************************************************************************")
    for tree in total_trees:
        print("Trees created using Random Forest algorithm are : ",tree)
        ans=ans+1
        start_time=time.time()
        rf=build_RF(training_data,m,tree,bootstrap)
        scratch_accuracy_list.append(cal_accuracy(testing_data,rf))
        end_time=time.time()
        rand_time+=(end_time-start_time)
        print("Executing time(in secs) for scratch code : " + str(round((end_time-start_time),2)))
        tree_list.append(tree)
        start_time=time.time()
        scikitlearn_accuracy_list.append(accuracy_using_scikitlearn(m,tree,training_data,testing_data))
        end_time=time.time()
        print("Executing time(in secs) for Scikit-learn code : " + str(round((end_time-start_time),2)))
        scikitlearn_time=(end_time-start_time)
        print("**************************************************************************")

if __name__ == "__main__":
    Random_Forest()
