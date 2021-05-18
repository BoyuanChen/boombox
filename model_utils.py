import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_only(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

class DebugLayer(nn.Module):
    def __init__(self):
        super(DebugLayer, self).__init__()
    
    def forward(self, x):
        import IPython
        IPython.embed()
        # assert False
        return x

class EncoderDecoder(torch.nn.Module):
    def __init__(self, in_channels, output_representation):
        super(EncoderDecoder,self).__init__()
        self.output_representation = output_representation
        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )

        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )

        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )
        
        self.deconv_7 = deconv_relu(128,128,4,stride=2)
        self.deconv_6 = deconv_relu(131,128,4,stride=2)
        self.deconv_5 = deconv_relu(131,128,4,stride=2)
        self.deconv_4 = deconv_relu(131,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        if self.output_representation == 'pixel':
            self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)
        if self.output_representation == 'segmentation':
            self.deconv_1 = deconv_relu(19,1,4,stride=2)
        if 'depth' in self.output_representation:
            self.deconv_1 = deconv_sigmoid(19,1,4,stride=2)

        self.predict_7 = torch.nn.Conv2d(128,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(131,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(131,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(131,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        
    def forward(self,x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)

        deconv7_out = self.deconv_7(conv7_out)
        predict_7_out = self.up_sample_7(self.predict_7(conv7_out))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))

        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)

        return predict_out