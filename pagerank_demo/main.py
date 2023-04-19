
import os
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
from tkinter.messagebox import *
from tkinter import scrolledtext
from Train import find_threshold
from Detection import detection

arg = []

def detect(f2):#检测部分
    global t1,t2,arg
    import input_CANBUSLOG
    test = input_CANBUSLOG.real_can(f2)# 读入检测数据
    print('2ing')
    detection(test,arg)#arg参数用于信息的传递，进程间变量的共享


def train1(f1):
    global t1,arg
    import input_CANBUSLOG
    train = input_CANBUSLOG.real_can(f1)#读入文件，不同的文件类型需要不同的读入预处理
    find_threshold(train,arg)#arg参数用于信息的传递，进程间变量的共享

def main():
    print("Ready for input")
    print("Select 1 for training\n Select 2 for detection")
    f=input()
    if(f=='1'):
        print("input file name")
        f1="./Attack_free_dataset1.txt"
        train1(f1)
    elif((f=='2')):
        print("input file name")
        f2="/Users/liufazhong/Downloads/SGX-CAN/SGX-CAN/dataset/OTIDS/Impersonation_attack_dataset.txt"
        print(f2)
        arg=detect(f2)
        print(arg)
    print("fin\n")
    return 0

if __name__ == '__main__' :
    main()