from myNeuron.layers_1 import FullyConnectedLayer, SigmoidLayer, SoftmaxLossLayer, ReLULayer
from myNeuron.layers_2 import ConvolutionalLayer, AveragePoolingLayer, FlattenLayer
import numpy as np
import time

class myCnn():
    def __init__(self,lr=0.0002):
        # self.param_path=param_path
        self.lr=lr
        self.param_layer_name = (
            'conv1_1', 'relu1_1', "pool1",'conv1_2', 'relu1_2', 'pool2',
            # 'flatten', 'fc4', 'sig4', 'fc5','sig5','fc6', 'softmax' #如果要使用sigmoid层请取注
            'flatten', 'fc4', 'relu4', 'fc5', 'relu5', 'fc6', 'softmax'

        )
        self.build_model()
        self.init_model()

    def build_model(self):
        # TODO：定义网络结构
        print('Building my model...')

        self.layers = {}
        self.layers['conv1_1'] = ConvolutionalLayer(3, 1, 4, 1, 1)
        self.layers['relu1_1'] = ReLULayer()
        self.layers["pool1"]=AveragePoolingLayer(2,2)
        self.layers['conv1_2'] = ConvolutionalLayer(3, 4, 8, 1, 1)
        self.layers['relu1_2'] = ReLULayer()
        self.layers['pool2'] = AveragePoolingLayer(2, 2)

        self.layers['flatten'] = FlattenLayer(input_shape=[8, 7, 7], output_shape=[392])

        # self.layers['fc4'] = FullyConnectedLayer (392, 128)
        # self.layers['sig4'] = SigmoidLayer()
        # self.layers['fc5'] = FullyConnectedLayer(128, 128)
        # self.layers['sig5'] = SigmoidLayer()
        # self.layers["fc6"]=FullyConnectedLayer(128,3)
        # self.layers['softmax'] = SoftmaxLossLayer()

        self.layers['fc4'] = FullyConnectedLayer (392, 128)
        self.layers['relu4'] = ReLULayer()
        self.layers['fc5'] = FullyConnectedLayer(128, 128)
        self.layers['relu5'] = ReLULayer()
        self.layers["fc6"]=FullyConnectedLayer(128,3)
        self.layers['softmax'] = SoftmaxLossLayer()


        self.update_layer_list = []
        for layer_name in self.layers.keys():
            if 'conv' in layer_name or 'fc' in layer_name:
                self.update_layer_list.append(layer_name)

    def init_model(self):
        print('Initializing parameters of each layer in myCnn...')
        for layer_name in self.update_layer_list:
            self.layers[layer_name].init_param()

    def forward(self):  # TODO：神经网络的前向传播
        # print('Inferencing...')
        start_time = time.time()
        current=self.input_image
        for idx in range(len(self.param_layer_name)):
            # print('Inferencing layer: ' + self.param_layer_name[idx])
            current = self.layers[self.param_layer_name[idx]].forward(current)
        # print('Inference time: %f' % (time.time()-start_time))
        return current

    def backward(self,loss):
        self.layers['softmax'].label_onehot = self.lable_onehot
        for idx in reversed(range(len(self.param_layer_name))):
            cur_layer_name=self.param_layer_name[idx]
            if cur_layer_name=="softmax":
                top_diff=self.layers[cur_layer_name].backward()
                self.lossVal=top_diff
            else:
                top_diff=self.layers[cur_layer_name].backward(top_diff)
                if "conv" in cur_layer_name or "fc" in cur_layer_name:
                    self.layers[cur_layer_name].update_param(self.lr)


    def evaluate(self,input,lable):
        # TODO：获取神经网络前向传播的结果
        self.input_image=input
        self.lable_onehot=lable
        prob = self.forward()
        # top1 = np.argmax(prob[0])
        # print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))
        return prob

    def predict(self,input,lable):
        # TODO：获取神经网络前向传播的结果
        self.input_image=input
        prob = self.forward()
        acc=0
        for i in range(input.shape[0]):
            if lable[i,np.argmax(prob[i])]==1:
                acc+=1
        acc/=input.shape[0]
        # print('Classification result: id = %d, prob = %f' % (top1, prob[0, top1]))
        return acc

    def loss(self):
        return self.layers['softmax'].get_loss(self.lable_onehot)

if __name__ == '__main__':
    model = myCnn("mnist012.npz")
    model.build_model()
    model.init_model()

    prob = model.evaluate()