#code
#coding=UTF-8
# ! -*- coding: utf-8 -*-
from __future__ import print_function
import tkinter as tk
from keras.models import load_model
import tkinter.ttk
import numpy as np
import matplotlib.pyplot as plt

import math
import networkx as nx

picn=0
def pr_init(data):
    global v
    global d1,d2
    v=set(data)
    d1={} # key:CANid Value:logic order
    d2={} # key:logic order Value:CANid
    for i in range(len(list(v))):
        d1[list(v)[i]]=i
        d2[str(i)]=list(v)[i]
# 建图 此处可改
def G_gen(edge):
    global v
    global d1,d2
    weight=[]
    G=nx.DiGraph()
    G.add_edges_from(edge)
    #nx:图处理包
    # 简单pagerank G是tu
    return nx.pagerank(G)

# 展示图片
def G_gen_pic(edge,d1,d2,v):#生成对应图片，在这里存图
    global picn
    weight=[]
    picn+=1
    G=nx.DiGraph()
    G.add_edges_from(edge)
    pos=nx.layout.spring_layout(G)
    node_sizes = [3+10* i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig("D:\\pic\\"+str(picn)+".png")
    plt.clf()
    return nx.pagerank(G)

# 生成向量，分割图
def pr_v(time,data,d1,d2,v,window=100):
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
                edge.append((data[i],data[i+1]))
        prl=G_gen(edge)
        for i in range(len(vec)):
            if d2[str(i)] in prl.keys():
                vec[i]=prl[d2[str(i)]]
            else:
                vec[i]=0
        ans.append(vec)
    return ans

# 专门用作展示，生成图像的pr_v
def pr_v_d(time,data,d1,d2,v,window=100):
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
                edge.append((data[i],data[i+1]))
        prl=G_gen_pic(edge,d1,d2,v)
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

# 载入模型
def detection(data_te,arg,time_window=100):#进行检测，参数为检测数据，时间窗口，编码器，生成器，阈值下界，阈值上界，batch大小，消息列表，CAN ID字典1和2以及ID集合
    #global message, root, ermsg, t2
    res=0
    total_num=0
    print(1)
    encoder=load_model("./enc.h5",compile=False)
    generator=load_model("./gen.h5",compile=False)

    with open("./threshold.txt","r") as f:
        tmp=f.readlines()
    f.close()
    th1=eval(tmp[0])
    th2=eval(tmp[1])
    batch_size=eval(tmp[2])
    import json
    with open("./d1.json","r") as f:
        tmp=f.read()
    f.close()
    d1=json.loads(tmp)
    with open("./d2.json","r") as f:
        tmp=f.read()
    f.close()
    d2=json.loads(tmp)
    with open("./v.json","r") as f:
        v=f.readlines()
    f.close()
    for i in range(len(v)):
        v[i]=v[i][:-1]
    cur = []
    cur_t = []
    # window = 5
    # m = [[0 for i in range(len(v))] for j in range(len(v))]
    flag = 0
    err = []
    err_1 = []
    err_2 = []
    new_ecu_err=[]
    pc_input = []
    for i in range(3):
        arg.append("")
    arg[0]=arg[0]+"开始检测！\n"
    # loop

    # 重放攻击特殊检测

    flag=0
    err_ls=[]
    tmp=[]
    for i in range(len(data_te["ID"])):
        tmp.append(data_te["time"][i])
        if data_te["data"][i]=='0102030405060708':
            flag=1
        if len(tmp) != 0 and len(tmp) % 100 == 0 and flag==1:
            err_ls.append(i)
            flag=0


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
                    new_ecu_err.append(i)
            pc_t=np.array(pr_v(cur_t,cur,d1,d2,v))
            pc_t = pc_t.reshape((len(pc_t), np.prod(pc_t.shape[1:])))
            pc_latent = encoder.predict(pc_t, batch_size=batch_size,verbose=0)
            x = generator.predict(pc_latent,verbose=0)
            xrc = get_mean(x[0])
            xt = get_mean(pc_t[0])
            rc_loss = -(xt * math.log(1e-10 + xrc) + (1 - xt) * math.log(1 - xrc))
            rc_loss = rc_loss  # *1e17%1e10
            print(rc_loss)
            theta = (th2 - th1) * 0.05
            # if (rc_loss < th1 or rc_loss > th2):
            if (rc_loss < th1 + theta or rc_loss > th2 - theta):
                err.append(i)
            theta = (th2 - th1) * 0.2
            if (rc_loss < th1 + theta or rc_loss > th2 - theta):
                err_1.append(i)
            theta = (th2 - th1) * 0
            if (rc_loss < th1 + theta or rc_loss > th2 - theta):
                err_2.append(i)
            pc_input = []
            cur = []
            cur_t = []

    data_len = len(data_te["ID"])

    a = set(err)
    # print(a)
    b = set(err_ls)
    # print(a & b)
    res = "检测结果:在delta=0.05，共" + str(len(data_te["ID"])) + "条CAN信息共" + str(
        int(len(data_te["ID"]) / time_window)) + "个向量中共检测出" + str(len(err)) + "条疑似入侵检测消息"

    print(len(a&b) / len(b))
    print(res)
    a = set(err_1)
    # print(a)
    # print(a & b)
    res = "检测结果:在delta=0.2,共" + str(len(data_te["ID"])) + "条CAN信息共" + str(
        int(len(data_te["ID"]) / time_window)) + "个向量中共检测出" + str(len(err_1)) + "条疑似入侵检测消息"
    print(len(a&b) / len(b))

    a = set(err_2)
    # print(a)
    # print(a&b)
    print(res)
    res = "检测结果:在delta=0，共" + str(len(data_te["ID"])) + "条CAN信息共" + str(
        int(len(data_te["ID"]) / time_window)) + "个向量中共检测出" + str(len(err_2)) + "条疑似入侵检测消息"
    print(len(a&b) / len(b))
    print(res)