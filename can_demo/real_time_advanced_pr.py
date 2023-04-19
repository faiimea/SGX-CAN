'''用Keras实现的VAE
   目前只保证支持Tensorflow后端
   改写自
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.datasets import mnist
#import main
import real_can
import graph_gen
from correlation import pearson,cosi
import input_data
import math
import networkx as nx
import input_data
from graph_gen import get_set,graph_gen

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



def find_threshold(data_tr,data_te,time_window=20):
    pr_init(data_tr["ID"])
    batch_size = 1
    original_dim=len(v)
    latent_dim = 2  # 隐变量取2维只是为了方便后面画图
    intermediate_dim = 10
    epochs = 1

    x = pr_v(data_tr["time"], data_tr["ID"], 20)
    # data={}
    # data["ID"]=data_train
    # v,v_set,e_set=graph_gen.get_set(data)
    # # data=input_data.input_data("Attack_free_dataset.txt")
    # # data_t=input_data.input_data("Impersonation_attack_dataset.txt")
    # data=main.draw(data,time_window)
    # x=[]
    # x_t=[]
    # #x_t=x
    # for i in range(0,int(len(data)/window)):
    #     x.append(data[i*window:(i+1)*window])
    # print(x)
    ooo=x
    x_train=np.array(x)
    #x_test=np.array(x_t)
    print(x_train)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    #x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation='relu')(x)

    # 算p(Z|X)的均值和方差
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)


    # 重参数技巧
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon


    # 重参数层，相当于给输入加入噪声
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码层，也就是生成器部分
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # 建立模型
    vae = Model(x, x_decoded_mean)

    # xent_loss是重构loss，kl_loss是KL loss
    xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_train, None))

    # 构建encoder，然后观察各个数字在隐空间的分布
    encoder = Model(x, z_mean)

    #x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

    x_encoded = encoder.predict(x_train, batch_size=batch_size)

    # 构建生成器
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    rc_test = []
    # x_rc_0 = generator.predict(x_encoded)
    # for i in range(len(x_train)):
    #     xrc = get_mean(x_rc_0[i])
    #     xt = get_mean(ooo[i])
    #     rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
    #     rc_loss=rc_loss#*1e17%1e11
    #     print(rc_loss)
    #     rc_test.append(rc_loss)
    from collections import Counter
    cur = []
    cur_t = []
    print("Train:")
    for i in range(len(data_tr["ID"])):
        #print(i)
        cur_t.append(data_tr["time"][i])
        cur.append(data_tr["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            print(i)
            cur_set = list(set(cur))
            for u in cur_set:
                if u in v:
                    continue
                else:
                    print(u)
            pc_t = np.array(pr_v(cur_t, cur))
            # print(pc_t.shape)
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            pc_latent = encoder.predict(pc_t, batch_size=batch_size)
            x = generator.predict(pc_latent)
            xrc = get_mean(x[0])
            xt = get_mean(pc_t[0])
            rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
            rc_loss = rc_loss #* 1e17 % 1e10
            rc_test.append(rc_loss)
            print(rc_loss)
        if(i<5030 and i>5000):
            th1, th2 = find(rc_test)
            encoder.save("./enc.h5")
            generator.save("./gen.h5")
            with open("./threshold.txt", "w") as f:
                f.write(str(th1) + "\n" + str(th2) + "\n" + str(batch_size))
            f.close()
            import json
            with open("./d1.json", "w") as f:
                f.write(json.dumps(d1))
            f.close()
            with open("./d2.json", "w") as f:
                f.write(json.dumps(d2))
            f.close()
            with open("./v.json", "w") as f:
                for i in v:
                    f.write(str(i) + "\n")
            f.close()
            break
    th1, th2 = find(rc_test)
    # 存储模型及相关参数
    encoder.save("./enc.h5")
    generator.save("./gen.h5")
    with open("./threshold.txt", "w") as f:
        f.write(str(th1) + "\n" + str(th2) + "\n" + str(batch_size))
    f.close()
    import json
    with open("./d1.json", "w") as f:
        f.write(json.dumps(d1))
    f.close()
    with open("./d2.json", "w") as f:
        f.write(json.dumps(d2))
    f.close()
    with open("./v.json", "w") as f:
        for i in v:
            f.write(str(i) + "\n")
    f.close()
    STR = "Store in" + i + "times\n"
    print(STR)

    print(Counter(rc_test))
    th1, th2 = find(rc_test)
    print(th1)
    print(th2)
    # data = {}#real_can.real_can("CAN bus log - no injection of messages.txt")


    cur = []
    cur_t = []
    # window = 5
    # m = [[0 for i in range(len(v))] for j in range(len(v))]
    flag = 0
    err = []
    new_ecu_err=[]
    pc_input = []
    print("Test")
    for i in range(len(data_te["ID"])):
        #print(i)
        cur_t.append(data_te["time"][i])
        cur.append(data_te["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            cur_set=list(set(cur))
            for u in cur_set:
                if u in v:
                    continue
                else:
                    print(u)
                    new_ecu_err.append(i)
            pc_t=np.array(pr_v(cur_t,cur))
            #print(pc_t.shape)
            pc_t=pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            pc_latent = encoder.predict(pc_t, batch_size=batch_size)
            x = generator.predict(pc_latent)
            xrc = get_mean(x[0])
            xt = get_mean(pc_t[0])
            rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
            rc_loss=rc_loss#*1e17%1e10
            print(rc_loss)
            theta=(th2-th1)*0.05
            if (rc_loss < th1  or rc_loss > th2 ):
            #if (rc_loss < th1+theta or rc_loss > th2-theta):
                err.append(i)
                #print("Alert at CAN instruction" + str(i))
            pc_input = []
            cur = []
            cur_t=[]
    print(new_ecu_err)
    print(len(err))
    return err


import input_CANBUSLOG
data1=input_CANBUSLOG.real_can("./Train_data1.txt")
#print(data1)
#v=set(data1["ID"])
data2=input_CANBUSLOG.real_can("./Test_data1.txt")
#data2=data1
#data1=PageRank.pr_v(data1["ID"],100)
#data2=PageRank.pr_v(data2["ID"],100)
print(find_threshold(data1,data2))
#print(Test(data2))


def Test(data_te,time_window=20):
    cur = []
    cur_t = []
    # window = 5
    # m = [[0 for i in range(len(v))] for j in range(len(v))]
    flag = 0
    err = []
    new_ecu_err = []
    pc_input = []
    encoder = load_model("./enc.h5", compile=False)
    generator = load_model("./gen.h5", compile=False)

    with open("./threshold.txt", "r") as f:
        tmp = f.readlines()
    f.close()
    th1 = eval(tmp[0])
    th2 = eval(tmp[1])
    batch_size = eval(tmp[2])
    import json
    with open("./d1.json", "r") as f:
        tmp = f.read()
    f.close()
    d1 = json.loads(tmp)
    with open("./d2.json", "r") as f:
        tmp = f.read()
    f.close()
    d2 = json.loads(tmp)
    with open("./v.json", "r") as f:
        v = f.readlines()
    f.close()
    print("Test")
    for i in range(len(data_te["ID"])):
        # print(i)
        cur_t.append(data_te["time"][i])
        cur.append(data_te["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            cur_set = list(set(cur))
            for u in cur_set:
                if u in v:
                    continue
                else:
                    print(u)
                    new_ecu_err.append(i)
            pc_t = np.array(pr_v(cur_t, cur))
            # print(pc_t.shape)
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            pc_latent = encoder.predict(pc_t, batch_size=batch_size)
            x = generator.predict(pc_latent)
            xrc = get_mean(x[0])
            xt = get_mean(pc_t[0])
            rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
            rc_loss = rc_loss  # *1e17%1e10
            print(rc_loss)
            theta = (th2 - th1) * 0.05
            if (rc_loss < th1 or rc_loss > th2):
                # if (rc_loss < th1+theta or rc_loss > th2-theta):
                err.append(i)
                # print("Alert at CAN instruction" + str(i))
            pc_input = []
            cur = []
            cur_t = []
    print(new_ecu_err)
    print(len(err))
    return err