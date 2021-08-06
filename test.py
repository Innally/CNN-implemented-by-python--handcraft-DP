import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

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

    

def predict_test():
    data=np.load("mnist012.npz")
    test_x=data["test_x012"]
    test_y=data["test_y012"]
    # 如果需要继续训练，取消注释下面两行
    file= "ok_mymodle_relu.pkl"
    acc=0
    with open(file,"rb") as f:
        model=pk.load(f)
    all_test=test_y.shape[0]
    batch=64
    itertime=int(all_test/64+1)
    for iter in range(itertime):
        start=iter*batch
        end=start+batch
        data=loadData(test_x,start,end) # load data of batch size
        lable=loadLable(test_y,start,end)
        result=model.predict(data,lable)
        acc+=result
        print("the prediction is",acc/(iter+1))
    return acc/(iter+1)

def predict_test(model):
    data=np.load("mnist012.npz")
    test_x=data["test_x012"]
    test_y=data["test_y012"]
    # 如果需要继续训练，取消注释下面两行
    acc=0
    all_test=test_y.shape[0]
    batch=64
    itertime=int(all_test/64+1)
    for iter in range(itertime):
        start=iter*batch
        end=start+batch
        data=loadData(test_x,start,end) # load data of batch size
        lable=loadLable(test_y,start,end)
        result=model.predict(data,lable)
        acc+=result
        print("the prediction is",acc/(iter+1))
    return acc/(iter+1),model.lossVal

if __name__=="__main__":
    # 可视化
    # filename="info_train.pkl"
    filename="info_test.pkl"

    with open(filename,'rb') as f:
        info=pk.load(f)
    # loss=info["train_loss"]
    # acc=info['train_acc']
    loss=info["test_loss"]
    acc=info['test_acc']
    loss=np.array(loss)
    loss=np.reshape(loss,(-1,1))
    acc=np.array(acc)
    acc=np.reshape(acc,(-1,1))

    plt.plot(range(len(loss)),loss[0:len(loss)],c="r",label="loss")
    plt.xlabel("Sample point time")
    plt.ylabel("loss")
    plt.show()
    plt.plot(range(len(acc)),acc[0:len(loss)],c="b",label="acc")
    plt.xlabel("Sample point time")
    plt.ylabel("accuracy")
    plt.show()