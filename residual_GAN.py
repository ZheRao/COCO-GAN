import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.2)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data,1.0,0.2)
        torch.nn.init.constant_(m.bias.data,0)

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(0.05)
        self.dropout = nn.Dropout2d(0.2)
        self.final_pool = nn.AvgPool2d(kernel_size=8,stride=1)
        self.final_dense = nn.Linear(2048,1)
        self.flatten = nn.Flatten(start_dim=1)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3,stride=1,bias=False),#no change
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64,64,kernel_size=7,padding=3,stride=2,bias=False), # /2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05)
        )
        self.conv1 = self._normal_block(kernel=1,padding=0,stride=1,
                                        in_c=64,out_c=256)

        self.bottleneck1 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=64,out_c=256,middle_c=64)
        self.bottleneck2 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=256,out_c=256,middle_c=64)
        self.conv3 = self._normal_block(kernel=1,padding=0,stride=2,
                                        in_c=256,out_c=512)
        self.bottleneck3 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,2,1],
                                                  in_c=256,out_c=512,middle_c=128)
        self.bottleneck4 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=512,out_c=512,middle_c=128)
        self.conv5 = self._normal_block(kernel=1,padding=0,stride=2,
                                        in_c=512,out_c=1024)
        self.bottleneck5 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,2,1],
                                                  in_c=512,out_c=1024,middle_c=256)
        self.bottleneck6 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=1024,out_c=1024,middle_c=256)
        self.bottleneck6_2 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=1024,out_c=1024,middle_c=256)
        self.conv7 = self._normal_block(kernel=1,padding=0,stride=2,
                                        in_c=1024,out_c=2048)
        self.bottleneck7 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,2,1],
                                                  in_c=1024,out_c=2048,middle_c=512)
        self.bottleneck8 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=2048,out_c=2048,middle_c=512)
        self.bottleneck8_2 = self._bottleneck_block(kernel=[1,3,1],
                                                  padding=[0,1,0],
                                                  stride=[1,1,1],
                                                  in_c=2048,out_c=2048,middle_c=512)
        self.apply(weights_init)
        

    def _normal_block(self,kernel,padding,stride,in_c,out_c,):
        block = nn.Sequential(
            nn.Conv2d(in_c,out_c,
                      kernel_size=kernel,padding=padding,stride=stride,bias=False),
            nn.BatchNorm2d(out_c)
        )
        return block
    
    def _bottleneck_block(self,kernel,padding,stride,in_c,out_c,middle_c):
        block = nn.Sequential(
            nn.Conv2d(in_c,middle_c,kernel_size=kernel[0],
                      padding = padding[0],stride=stride[0],bias=False),
            nn.BatchNorm2d(middle_c),
            nn.LeakyReLU(0.05),
            nn.Conv2d(middle_c,middle_c,kernel_size=kernel[1],
                      padding = padding[1],stride=stride[1],bias=False),
            nn.BatchNorm2d(middle_c),
            nn.LeakyReLU(0.05),
            nn.Conv2d(middle_c,out_c,kernel_size=kernel[2],
                      padding = padding[2],stride=stride[2],bias=False),
            nn.BatchNorm2d(out_c),
        )
        return block
    
    def forward(self, x):
        output = self.initial_conv(x)
        output = self.dropout(output)
        resnet_1 = self.activation(self.conv1(output) + self.bottleneck1(output))
        resnet_1 = self.dropout(resnet_1)
        resnet_2 = self.activation(resnet_1 + self.bottleneck2(resnet_1))
        resnet_2 = self.dropout(resnet_2)
        resnet_3 = self.activation(self.conv3(resnet_2) + self.bottleneck3(resnet_2))
        resnet_3 = self.dropout(resnet_3)
        resnet_4 = self.activation(resnet_3+self.bottleneck4(resnet_3))
        resnet_4 = self.dropout(resnet_4)
        resnet_5 = self.activation(self.conv5(resnet_4) + self.bottleneck5(resnet_4))
        resnet_5 = self.dropout(resnet_5)
        resnet_6 = self.activation(resnet_5+self.bottleneck6(resnet_5))
        resnet_6 = self.dropout(resnet_6)
        resnet_6_2 = self.activation(resnet_6+self.bottleneck6_2(resnet_6))
        resnet_6_2 = self.dropout(resnet_6_2)
        resnet_7 = self.activation(self.conv7(resnet_6_2) + self.bottleneck7(resnet_6_2))
        resnet_7 = self.dropout(resnet_7)
        resnet_8 = self.activation(resnet_7+self.bottleneck8(resnet_7))
        resnet_8 = self.dropout(resnet_8)
        resnet_8_2 = self.activation(resnet_8+self.bottleneck8_2(resnet_8))
        resnet_8_2 = self.dropout(resnet_8_2)
        output = self.final_pool(resnet_8_2)
        output = self.flatten(output)
        output = self.final_dense(output)
        return output