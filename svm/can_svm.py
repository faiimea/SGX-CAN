import networkx as nx
from sklearn.svm import OneClassSVM
import input_CANBUSLOG

'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

import numpy as np
import joblib
import networkx as nx

# 创建有向图
v={}
d1={}
d2={}
def pr_init(data):
    global v
    global d1,d2
    v=set(data)
    d1={} # key:CANid Value:logic order
    d2={} # key:logic order Value:CANid
    for i in range(len(list(v))):
        d1[list(v)[i]]=i
        d2[str(i)]=list(v)[i]
def G_gen(edge):
    global v
    global d1,d2
    weight=[]
    G=nx.MultiGraph()
    G.add_weighted_edges_from(edge)
    return nx.pagerank(G)
def pr_v(time,data,window=20):
    global v
    global d1
    global d2
    #pr_init(data)
    #print(time)
    #print(data)
    ans=[]
    vec = []
    for i in list(v):
        vec.append(0)
    l=int((len(data))/window)
    for t in range(l):
        edge=[]
        #generate Pagerank
        for i in range(t*window,(t+1)*window):
            if (i+1)<len(data):
                # change time to float(time)
                edge.append((data[i],data[i+1],(float(time[i+1])-float(time[t*window]))))
        prl=G_gen(edge)
        for i in range(len(vec)):
            if d2[str(i)] in prl.keys():
                vec[i]=prl[d2[str(i)]]
            else:
                vec[i]=0
        ans.append(vec)
    return ans
def find(l):
    min0=min(l)
    max0=max(l)
    return min0,max0
def get_mean(l):
    s = 0
    for i in l:
        s = s + i
    return s / len(l)


def train_svm(data_tr,time_window=20):
    pr_init(data_tr["ID"])
    cur = []
    cur_t = []
    print("Train:")
    flag = 0
    print(len(data_tr["ID"]))
    for i in range(len(data_tr["ID"])):
        if i % 10000 == 0:
            print(i)
        cur_t.append(data_tr["time"][i])
        cur.append(data_tr["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            pc_t = np.array(pr_v(cur_t, cur))
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            if (flag == 0):
                train_data = pc_t
                flag = 1
            else:
                train_data = np.concatenate((train_data, pc_t), axis=0)
            cur = []
            cur_t = []

    flag = 0
    err = []
    clf = OneClassSVM(kernel='rbf', nu=0.1)  # 这里的参数可以根据具体情况进行调整
    clf.fit(train_data)
    joblib.dump(clf, 'model.pkl')

def test_svm(data_te,time_window=20):
    cur=[]
    cur_t=[]
    clf = joblib.load('model.pkl')

    flag=0


   # xxx=len(err_ls)
    flag=0

    for i in range(len(data_te["ID"])):
        cur_t.append(data_te["time"][i])
        cur.append(data_te["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            pc_t=np.array(pr_v(cur_t,cur))
            pc_t=pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            if (flag == 0):
                test_data =pc_t
                flag=1
            else:
                test_data=np.concatenate((test_data,pc_t),axis=0)
            cur = []
            cur_t=[]

    prediction=clf.predict(test_data)
    A_num=0
    B_num=0
    for i in prediction:
        if i == 1:
            A_num+=1
        elif i == -1:
            B_num+=1
    print(A_num)
    print(B_num)

def find_threshold(data_tr,data_te,time_window=20):
    pr_init(data_tr["ID"])
    cur = []
    cur_t = []
    print("Train:")
    flag=0
    err_ls = []
    tmp = []
    j = 0
    for i in range(len(data_te["ID"])):
        tmp.append(data_te["time"][i])
        if data_te["data"][i] == '0102030405060708':
            flag = 1
        if len(tmp) != 0 and len(tmp) % 20 == 0 and flag == 1:
            err_ls.append(j)
            flag = 0
        if len(tmp) != 0 and len(tmp) % 20 ==0:
            j+=1

    flag=0
    for i in range(len(data_tr["ID"])):
        cur_t.append(data_tr["time"][i])
        cur.append(data_tr["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            pc_t = np.array(pr_v(cur_t, cur))
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            if (flag == 0):
                train_data =pc_t
                flag=1
            else:
                train_data=np.concatenate((train_data,pc_t),axis=0)
            cur = []
            cur_t = []

    flag = 0
    err = []
    clf = OneClassSVM(kernel='rbf', nu=0.1)  # 这里的参数可以根据具体情况进行调整
    clf.fit(train_data)


    print("Test")
    for i in range(len(data_te["ID"])):
        cur_t.append(data_te["time"][i])
        cur.append(data_te["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            pc_t=np.array(pr_v(cur_t,cur))
            pc_t=pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            if (flag == 0):
                test_data =pc_t
                flag=1
            else:
                test_data=np.concatenate((test_data,pc_t),axis=0)
            cur = []
            cur_t=[]

    prediction=clf.predict(test_data)
    A_num=0
    B_num=0

    R_R=0
    R_E=0
    E_R=0
    E_E=0
    tmp_r=[]
    tmp_e=[]
    for i in range(len(prediction)):
        if(prediction[i]==1):
            tmp_r.append(i)
        else:
            tmp_e.append(i)
    tmp_e=set(tmp_e)
    tmp_r=set(tmp_r)
    err_ls=set(err_ls)
    print(len(tmp_r))
    print(len(tmp_e))
    print(len(err_ls))
    print(len(err_ls&tmp_e))
    # for i in prediction:
    #     if i == 1 and err_ls:
    #         A_num+=1
    #     elif i == -1:
    #         B_num+=1
    # print(A_num)
    # print(B_num)
    #print(prediction)
    return err


data1=input_CANBUSLOG.real_can("/Users/liufazhong/Downloads/SGX-CAN/SGX-CAN/dataset/OTIDS/Attack_free_dataset1.txt")
data2=input_CANBUSLOG.real_can("/Users/liufazhong/Downloads/SGX-CAN/SGX-CAN/dataset/OTIDS/Impersonation_attack_dataset.txt")

#train_svm(data1)
find_threshold(data1,data2)