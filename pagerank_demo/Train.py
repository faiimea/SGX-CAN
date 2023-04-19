#code
#coding=UTF-8
# ! -*- coding: utf-8 -*-
from __future__ import print_function
import time

import tkinter as tk
import tkinter.ttk
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

import math
import networkx as nx
from Detection import detection
v={}
d1={}
d2={}
def pr_init(data):#预处理CAN ID，为每个CAN ID生成一个对应的序号并存储
    global v
    global d1,d2
    v=set(data)
    d1={} # key:CANid Value:logic order
    d2={} # key:logic order Value:CANid
    for i in range(len(list(v))):
        d1[list(v)[i]]=i
        d2[str(i)]=list(v)[i]

def G_gen(edge):#生成有向图并计算pagerank值
    global v
    global d1,d2
    weight=[]
    G=nx.DiGraph()
    G.add_edges_from(edge)
    return nx.pagerank(G)

def pr_v(time,data,window=100):#切片并生成对应的向量，其中time为时间，data为数据，window是时间窗口
    global v
    global d1
    global d2
    ans=[]
    vec = []
    for i in list(v):
        vec.append(0)
    l=int((len(data))/window)
    for t in range(l):
        edge=[]
        #generate Pagerank
        for i in range(t*window,(t+1)*window):#划分并生成向量
            if (i+1)<len(data):
                edge.append((data[i],data[i+1]))
        prl=G_gen(edge)
        for i in range(len(vec)):
            if d2[str(i)] in prl.keys():
                vec[i]=prl[d2[str(i)]]
            else:
                vec[i]=0
        ans.append(vec)
    return ans

def find(l):#找到阈值
    min0=min(l)
    max0=max(l)
    return min0,max0
def get_mean(l):#获取平均值
    s = 0
    for i in l:
        s = s + i
    return s / len(l)

# VAE训练
def find_threshold(data_tr,arg,time_window=100):#训练，参数值为训练数据，检验数据，消息列表和时间窗口大小
    # 数据预处理
    # 注意格式对应的问题！！！
    for i in range(3):
        arg.append("")
    arg[0]="开始训练！\n"
    pr_init(data_tr["ID"])
    batch_size = 1
    original_dim=len(v)
    latent_dim = 2  # 隐变量取2维只是为了方便后面画图
    intermediate_dim = 10
    epochs = 10

    x = pr_v(data_tr["time"], data_tr["ID"], 100)
    x_train=np.array(x)
    print(x_train)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

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
    arg[0]=arg[0]+"模型建立中！\n"
    vae = Model(x, x_decoded_mean)

    # xent_loss是重构loss，kl_loss是KL loss
    xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    ##完成神经网络构建 f


    # region tk

    arg[0] =arg[0] + "初始化完毕！\n"


    # endregion

    # 模型训练 vae.fit

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_train, None))



    # 确定阈值阶段
    # 通过已经训练的模型再次运行数据
    # 构建encoder，然后观察各个数字在隐空间的分布
    encoder = Model(x, z_mean)

    x_encoded = encoder.predict(x_train, batch_size=batch_size)

    # 构建生成器
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    print("ok")
    rc_test = []
    # region tk
    from collections import Counter
    cur = []
    cur_t = []
    arg[0]=arg[0]+"阈值划定...\n"
    # endregion
    for i in range(len(data_tr["ID"])):
        cur_t.append(data_tr["time"][i])
        cur.append(data_tr["ID"][i])
        if len(cur) != 0 and len(cur) % time_window == 0:
            cur_set = list(set(cur))
            for u in cur_set:
                if u in v:
                    continue
                else:
                    print(u)
            pc_t = np.array(pr_v(cur_t, cur))
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            pc_latent = encoder.predict(pc_t, batch_size=batch_size,verbose=0)
            x = generator.predict(pc_latent,verbose=0)
            xrc = get_mean(x[0])
            xt = get_mean(pc_t[0])
            rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
            rc_loss = rc_loss #* 1e17 % 1e10
            rc_test.append(rc_loss)
        if i%500000==0 and i>0:
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
            STR="Store in"+i+"times\n"
            print(STR)
    th1, th2 = find(rc_test)

    arg[0]=arg[0]+"训练完成！\n阈值下界: "+str(th1)+"\n阈值上界: "+str(th2)+"\n"

    # 存储模型及相关参数

    encoder.save("./enc.h5")
    generator.save("./gen.h5")
    with open("./threshold.txt","w") as f:
        f.write(str(th1)+"\n"+str(th2)+"\n"+str(batch_size))
    f.close()
    import json
    with open("./d1.json","w") as f:
        f.write(json.dumps(d1))
    f.close()
    with open("./d2.json","w") as f:
        f.write(json.dumps(d2))
    f.close()
    with open("./v.json","w") as f:
        for i in v:
            f.write(str(i)+"\n")
    f.close()
    print("Train Done")
