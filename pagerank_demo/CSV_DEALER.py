import csv
# 处理输入
def csvcan(file):
    res={}
    flag=1
    ans=[]
    time=[]
    with open(file)as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            if len(row)>10 and row[9]=="PT":
                flag=0
                continue
            if flag==0:
                if row[6]!="CAN Bus Event":
                    time.append(eval(row[1]))
                    ans.append(row[9])
    res["ID"]=ans
    res["time"]=time
    return res
#print(csvcan("demo/1.csv")["ID"])