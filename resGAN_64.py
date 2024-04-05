import torch
import torch.nn as nn

class Discriminator(torch.nn.Module):
    def __init__(self, minibatch_features = 128, features_dim=8):
        super().__init__()
        self.features = minibatch_features
        self.features_dim = features_dim
        self.activation = nn.LeakyReLU(0.05)
        self.dropout = nn.Dropout2d(0.2)
        self.final_pool = nn.AvgPool2d(kernel_size=8,stride=1)
        self.flatten = nn.Flatten(start_dim=1)
        self.minibatch_linear = nn.Linear(1024,self.features*self.features_dim)
        self.final_dense = nn.Linear(1024+2*self.features,1)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3,stride=1,bias=False),#no change
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.05),
            nn.Conv2d(64,64,kernel_size=7,padding=3,stride=2,bias=False), # /2
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05)
        )
        self.conv1 = self._normal_block(kernel=1,padding=0,stride=1,in_c=64,out_c=256)

        self.bottleneck1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                                  in_c=64,out_c=256,middle_c=64)
        
        self.bottleneck2 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                                  in_c=256,out_c=256,middle_c=64)
        
        self.conv3 = self._normal_block(kernel=1,padding=0,stride=2,in_c=256,out_c=512)
        
        self.bottleneck3 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,2,1],
                                                  in_c=256,out_c=512,middle_c=128)
        
        self.bottleneck4 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                                  in_c=512,out_c=512,middle_c=128)
        
        self.conv5 = self._normal_block(kernel=1,padding=0,stride=2,in_c=512,out_c=1024)
        
        self.bottleneck5 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,2,1],
                                                  in_c=512,out_c=1024,middle_c=256)
        
        self.bottleneck6 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                                  in_c=1024,out_c=1024,middle_c=256)


        
        #self.apply(weights_init)
        

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
    
    def _minibatch_discrimination(self,x,n_feature,feature_dim):
        batch_size = x.shape[0]
        x_copy = x.clone()
        x = self.minibatch_linear(x) # e.g., (4, 6)
        #print(f"after linear layer x shape: {x.shape}")
        x = x.view(-1,n_feature,feature_dim) # e.g., (4,3,2)
        #print(f"after reshape x shape: {x.shape}")
        # create mask
        mask = torch.eye(batch_size) # (4,4)
        mask = mask.unsqueeze(1) # (4, 1, 4)
        mask = (1 - mask).to('cuda')
        # calculate diff between features: goal (4, 3, 4)
        m1 = x.unsqueeze(3) # (4,3 2, 1)
        m2 = x.transpose(0,2).transpose(0,1).unsqueeze(0) # (1, 3, 2, 4)
        diff = torch.abs(m1 - m2) # (4, 3, 2, 4)
        diff = torch.sum(diff, dim=2) # (4, 3, 4)
        diff = torch.exp(-diff)
        diff_masked = diff * mask
        #print(f"diff_masked shape {diff_masked.shape}")
        # split sum up the differences goal (4,3*2)
        def half(tensor,second):
            return tensor[:,:,second*batch_size//2:(second+1)*batch_size//2]
        first_half = half(diff_masked, 0) # (4, 3, 2)
        first_half = torch.sum(first_half, dim=2) / torch.sum(first_half) # (4, 3)
        second_half = half(diff_masked, 1) 
        second_half = torch.sum(second_half, dim=2) / torch.sum(second_half)
        features = torch.cat([first_half,second_half], dim=1) # (4, 3*2)
        #print(f"features shape {features.shape}")
        # merge back to the input, goal (4,3*2*2)
        output = torch.cat([x_copy,features], dim=1)
        #print(output.shape)
        return output

    
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
        output = self.final_pool(resnet_6)
        output = self.flatten(output)
        print(output.shape)
        output = self._minibatch_discrimination(output,self.features,self.features_dim)
        output = self.final_dense(output)
        return output


class Generator(nn.Module):
    def __init__(self, noise_dim, final_activation="sigmoid"):
        super().__init__()
        self.noise_dim = noise_dim
        self.activation = nn.LeakyReLU(0.05)
        self.dropout = nn.Dropout2d(0.2)
        self.initial_convT = nn.ConvTranspose2d(1024,2048,kernel_size=8,padding=0)
        self.bottleneck8_3 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=2048, out_c=2048, middle_c=2048)
        self.bottleneck8_2 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=2048, out_c=2048, middle_c=2048)
        self.bottleneck8_1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=2048, out_c=2048, middle_c=2048)
        self.conv7 = self._normal_block(kernel=2,padding=0,stride=2,in_c=2048,out_c=1024)
        self.bottleneck7 = self._bottleneck_block(kernel=[1,2,1],padding=[0,0,0],stride=[1,2,1],
                                              in_c=2048, out_c=1024, middle_c=1024)
        self.bottleneck6_3 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=1024, out_c=1024, middle_c=1024)
        self.bottleneck6_2 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=1024, out_c=1024, middle_c=1024)
        self.bottleneck6_1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=1024, out_c=1024, middle_c=1024)
        self.conv5 = self._normal_block(kernel=2,padding=0,stride=2,in_c=1024,out_c=512)
        self.bottleneck5 = self._bottleneck_block(kernel=[1,2,1],padding=[0,0,0],stride=[1,2,1],
                                              in_c=1024, out_c=512, middle_c=512)
        self.bottleneck4_2 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=512, out_c=512, middle_c=512)
        self.bottleneck4_1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=512, out_c=512, middle_c=512)
        self.conv3 = self._normal_block(kernel=2,padding=0,stride=2,in_c=512,out_c=256)
        self.bottleneck3 = self._bottleneck_block(kernel=[1,2,1],padding=[0,0,0],stride=[1,2,1],
                                              in_c=512, out_c=256, middle_c=256)
        self.bottleneck2_2 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=256, out_c=256, middle_c=256)
        self.bottleneck2_1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=256, out_c=256, middle_c=256)
        self.conv1 = self._normal_block(kernel=1,padding=0,stride=1,in_c=256,out_c=64)
        self.bottleneck1 = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=256, out_c=64, middle_c=64)
        self.final_bottleneck = self._bottleneck_block(kernel=[1,3,1],padding=[0,1,0],stride=[1,1,1],
                                              in_c=64, out_c=32, middle_c=32)
        self.final_conv = self._normal_block(kernel=1,padding=0,stride=1,in_c=64,out_c=32)
        self.final_layer = nn.ConvTranspose2d(32,3,1,1,0)
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Tanh()
        #self.apply(weights_init)
        

    
    def _bottleneck_block(self,kernel,padding,stride,in_c,out_c,middle_c):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_c,middle_c,kernel_size=kernel[0],
                               padding=padding[0],stride=stride[0],bias=False),
            nn.BatchNorm2d(middle_c),
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(middle_c,middle_c,kernel_size=kernel[1],
                               padding=padding[1],stride=stride[1],bias=False),
            nn.BatchNorm2d(middle_c),
            nn.LeakyReLU(0.05),
            nn.ConvTranspose2d(middle_c,out_c,kernel_size=kernel[2],
                               padding=padding[2],stride=stride[2],bias=False),
            nn.BatchNorm2d(out_c),
        )
        return block
    def _normal_block(self,kernel,padding,stride,in_c,out_c):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,kernel_size=kernel,
                               padding=padding,stride=stride,bias=False),
            nn.BatchNorm2d(out_c),
        )
        return block

    
    def forward(self,x):
        B, L = x.shape
        assert L == self.noise_dim, f"noise dimension should be {self.noise_dim}"
        x = x.view(B,L,1,1)
        x = self.initial_convT(x)
        resnet8_3 = self.activation(x + self.bottleneck8_3(x))
        resnet8_3 = self.dropout(resnet8_3)
        resnet8_2 = self.activation(resnet8_3 + self.bottleneck8_2(resnet8_3))
        resnet8_2 = self.dropout(resnet8_2)
        resnet8_1 = self.activation(resnet8_2 + self.bottleneck8_1(resnet8_2))
        resnet8_1 = self.dropout(resnet8_1)
        resnet7 = self.activation(self.conv7(resnet8_1)+self.bottleneck7(resnet8_1))
        resnet7 = self.dropout(resnet7)
        resnet6_3 = self.activation(resnet7 + self.bottleneck6_3(resnet7))
        resnet6_3 = self.dropout(resnet6_3)
        resnet6_2 = self.activation(resnet6_3 + self.bottleneck6_2(resnet6_3))
        resnet6_2 = self.dropout(resnet6_2)
        resnet6_1 = self.activation(resnet6_2 + self.bottleneck6_1(resnet6_2))
        resnet6_1 = self.dropout(resnet6_1)
        resnet5 = self.activation(self.conv5(resnet6_1)+self.bottleneck5(resnet6_1))
        resnet5 = self.dropout(resnet5)
        resnet4_2 = self.activation(resnet5 + self.bottleneck4_2(resnet5))
        resnet4_2 = self.dropout(resnet4_2)
        resnet4_1 = self.activation(resnet4_2 + self.bottleneck4_1(resnet4_2))
        resnet4_1 = self.dropout(resnet4_1)
        resnet3 = self.activation(self.conv3(resnet4_1)+self.bottleneck3(resnet4_1))
        resnet3 = self.dropout(resnet3)
        resnet2_2 = self.activation(resnet3 + self.bottleneck2_2(resnet3))
        resnet2_2 = self.dropout(resnet2_2)
        resnet2_1 = self.activation(resnet2_2 + self.bottleneck2_1(resnet2_2))
        resnet2_1 = self.dropout(resnet2_1)
        resnet1 = self.activation(self.conv1(resnet2_1)+self.bottleneck1(resnet2_1))
        resnet1 = self.dropout(resnet1)
        output = self.activation(self.final_conv(resnet1) + self.final_bottleneck(resnet1))
        output = self.final_layer(output)
        output = self.final_activation(output)
        return output