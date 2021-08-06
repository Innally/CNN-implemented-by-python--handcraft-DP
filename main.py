import numpy as np
import cv2 as cv
from myCnn import myCnn
import time
from math import ceil
import pickle
from test import predict_test


def train(x,y,lr,batch_size,epoch):
    file = "mymodle.pkl"
    info_trian_file = 'info_train.pkl'
    info_test_file = 'info_test.pkl'

    # 如果需要继续训练，取消注释下面两行
    # with open(file,"rb") as f:
    #     model=pickle.load(f)

    # 如果需要重新开始训练，取消注释
    model=myCnn(lr=lr)
    model.init_model()

    all_num=x.shape[0] # 训练集所有图片数量
    train_info={}
    test_info={}
    train_info['train_acc']=[]
    train_info['train_loss']=[]
    test_info['test_acc']=[]
    test_info['test_loss']=[]
    if all_num%batch_size!=0:
        iter_time=ceil(all_num/batch_size)
    else :
        iter_time= int(all_num/batch_size)
        print("start training:")
    for ep in range(epoch):
        iter=0
        clock=time.time()
        acc=0
        for i in range(iter_time):
            start=iter*batch_size
            end=start+batch_size
            data=loadData(x,start,end) # load data of batch size
            lable=loadLable(y,start,end)
            result=model.evaluate(data,lable)  # here evaluate called forward to train
            loss=model.loss()
            percentage =ceil(iter / iter_time * 50)
            print("training process of epoch", ep, "is [", "#" * percentage, " " * (50 - percentage), "]",
                  round(iter/iter_time*100,2),"%","loss ==",loss)
            model.backward(loss) # backward to update the parameters

            # to get the acc number of this batch
            epc_pre=0
            for i in result:
                if y[iter*batch_size+epc_pre,np.argmax(i)]==1:
                    acc+=1
                epc_pre+=1
            iter+=1

            if iter%10==0:
                print("++++++now it's the %d iterations of training, and the acc== %f +++++++++"
                      %(iter % batch_size+(iter*iter_time), acc / end))
                train_info["train_acc"].append(acc / end)
                train_info["train_loss"].append(loss)

            if end%100==0:
                test_acc,test_loss=predict_test(model)
                test_info["test_acc"].append(test_acc)
                test_info["test_loss"].append(test_loss)

            clock_end=time.time()
            if iter%50==0:
                with open(file,"wb") as f:
                    pickle.dump(model, f)
        # 存储训练信息，用于可视化
        train_info["train_acc"].append(acc/all_num)
        train_info["train_loss"].append(loss)
        with open(info_trian_file,'wb') as f:
            pickle.dump(train_info,f)
        with open(info_test_file,'wb') as f:
            pickle.dump(test_info,f)
    
        # 输出时间
        epoch_time=(clock_end-clock)/3600
        print("Epoch %d has finished, it took %f hour(s), and the acc==%f"%(iter%batch_size,epoch_time,acc/all_num))
    return

def loadData(x,start,end):
    batch=[]
    if end>=x.shape[0]:
        end=x.shape[0]
    for idx_pic in range(start, end):
        batch.append(x[idx_pic])
    batch = np.array(batch).reshape((len(batch), 1, 28, 28))
    return batch
def loadLable(y,start,end):
    batch=[]
    if end>=y.shape[0]:
        end=y.shape[0]
    for idx_lable in range(start, end):
        batch.append(y[idx_lable])
    batch = np.array(batch).reshape((len(batch),3))
    return batch

if __name__=="__main__":
    # 导入预处理的数据
    data=np.load("mnist012.npz")
    test_x=data["test_x012"]
    test_y=data["test_y012"]
    train_x=data["train_x012"]
    train_y=data["train_y012"]
    lr=0.0001
    batch_size=64
    epoch=5
    train(train_x,train_y,lr,batch_size,epoch)

    # 初始化模型