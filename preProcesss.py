import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np



if __name__=="__main__":
    # import the datasets
    # train.images: the images
    # train.labels: lable of the images;
    train=mnist.train
    test=mnist.test
    xtr=train.images
    ytr=train.labels
    xtest=test.images
    ytest=test.labels
    train_x012=[]
    train_y012=[]
    test_x012=[]
    test_y012=[]
    for x,y in zip(xtr,ytr):
        if y[0]==1:
            train_x012.append(x)
            train_y012.append(np.array((1,0,0)))
        if y[1]==1:
            train_x012.append(x)
            train_y012.append(np.array((0,1,0)))
        if y[2]==1:
            train_x012.append(x)
            train_y012.append(np.array((0,0,1)))
    for x,y in zip(xtest,ytest):
        if y[0]==1:
            test_x012.append(x)
            test_y012.append(np.array((1,0,0)))
        if y[1]==1:
            test_x012.append(x)
            test_y012.append(np.array((0,1,0)))
        if y[2]==1:
            test_x012.append(x)
            test_y012.append(np.array((0,0,1)))

    train_y012=np.array(train_y012)
    test_y012=np.array(test_y012)
    test_x012=np.array(test_x012)
    np.savez("mnist012",train_x012=train_x012,train_y012=train_y012,test_x012=test_x012,test_y012=test_y012)

    print("End of the prediction",xtr[0])