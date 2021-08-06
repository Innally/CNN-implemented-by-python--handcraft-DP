# coding=utf-8
import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.3):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        # TODO: 边界扩充
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.input.shape[2]+self.padding, self.padding:self.input.shape[2]+self.padding] = self.input
        height_out = int((height - self.kernel_size) / self.stride +1)
        width_out = int((width - self.kernel_size) / self.stride + 1)

        self.output = np.zeros([self.input.shape[0], self.channel_out, int(height_out), int(width_out)])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]) + self.bias[idxc]
        return self.output

    def backward(self,top_diff):
        # TODO: 边界扩充
        height_out = self.input.shape[2]
        width_out = self.input.shape[3]
        height = top_diff.shape[2] + self.padding * 2
        width = top_diff.shape[3] + self.padding * 2
        top_diff_pad = np.zeros([self.input.shape[0], self.channel_out, int(height), int(width)])
        top_diff_pad[:, :, self.padding:top_diff.shape[2]+self.padding, self.padding:top_diff.shape[2]+self.padding] = top_diff
        self.bottom_diff = np.zeros([self.input.shape[0], self.channel_in, int(height_out), int(width_out)])

        # TODO:先求 bottom_diff
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_in):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        weight=np.rot90(np.rot90(self.weight))
                        self.bottom_diff[idxn, idxc, idxh, idxw] = np.sum(np.transpose(weight[idxc, :, :, :],(2,0,1))* top_diff_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
        w_height_out=self.weight.shape[1]
        w_width_out=self.weight.shape[2]
        self.d_weight=np.zeros_like(self.weight)
        top_diff_kernel_size=top_diff.shape[2]
        self.input_pad=np.transpose(self.input_pad,(1,0,2,3))
        # TODO: 再求dw
        for idxn in range(weight.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(w_height_out):
                    for idxw in range(w_width_out):
                        self.d_weight[idxn, idxh, idxw,idxc] = np.sum(top_diff[:, idxc, :, :] * self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+top_diff_kernel_size, idxw*self.stride:idxw*self.stride+top_diff_kernel_size])
        # TODO: 求出b
        self.d_bias = np.sum(top_diff,axis=(0,2,3))
        return self.bottom_diff

    def update_param(self, lr):  # 参数更新
        # TODO：对卷积层参数利用参数进行更新
        self.weight = self.weight - lr * self.d_weight
        self.bias = self.bias - lr * self.d_bias

    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

class AveragePoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 平均池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = int((self.input.shape[2] - self.kernel_size) / self.stride + 1)
        width_out = int((self.input.shape[3] - self.kernel_size) / self.stride + 1)
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算平均池化层的前向传播， 取池化窗口内的平均值
                        self.output[idxn, idxc, idxh, idxw] = np.mean(self.input[idxn,idxc,idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride + self.kernel_size])
        return self.output

    def backward(self, top_diff):
        self.bottom_diff=np.zeros_like(self.input)
        height_in=self.input.shape[2]
        width_in = self.input.shape[3]
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_in):
                    for idxw in range(width_in):
                        self.bottom_diff[idxn, idxc, idxh:idxh+self.kernel_size, idxw:idxw+self.kernel_size]+=top_diff[idxn,idxc,int(idxh/self.kernel_size),int(idxw/self.kernel_size)]/(self.kernel_size**2)
        return self.bottom_diff


class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        # 扁平化层的初始化
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):  # 前向传播的计算
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        # TODO：转换 input 维度顺序
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output

    def backward(self,top_diff):
        # TODO: 拉回原来的维度
        self.bottom_diff= np.transpose(np.reshape(top_diff,self.input.shape),(0,3,1,2))
        return self.bottom_diff
