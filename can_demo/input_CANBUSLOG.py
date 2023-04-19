import os
import sys
def real_can(FILE):
    # DATA={}
    # with open(FILE,"r") as f:
    #     raw_data=f.readlines()
    # ID=[]
    # time=[]
    # data=[]
    # for i in range(len(raw_data)):
    #     elem=raw_data[i].split()
    #     time.append(eval(elem[0][1:-1]))
    #     ID.append(elem[2].split("#")[0])
    #     data.append(elem[2].split("#")[1])
    # DATA["time"]=time
    # DATA["ID"]=ID
    # DATA["data"]=data
    #
    # return DATA
    DATA = {}
    with open(FILE, "r") as f:
        raw_data = f.readlines()
    Ts = []
    ID = []
    Tmp = []
    DLC = []
    data = []
    for i in range(len(raw_data)):
        elem = raw_data[i].split()
        Ts.append(elem[1])
        ID.append(elem[3])
        Tmp.append(elem[4])
        DLC.append(elem[6])
        n = eval(elem[6])
        s = ""
        if n > 0:
            for j in range(n):
                s = s + elem[7 + j]
        data.append(s)
    DATA["time"] = Ts
    DATA["ID"] = ID
    DATA["Tmp"] = Tmp
    DATA["DLC"] = DLC
    DATA["data"] = data
    return DATA