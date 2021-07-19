__author__ = 'yunbo'

import torch
import torch.nn as nn
import numpy as np
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(ConvLSTMCell, self).__init__()
        print("in_channel=",in_channel,"num_hidden=",num_hidden,"width=",width,"layer_norm=",layer_norm)
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size = filter_size, stride = stride, padding=self.padding),
            # nn.LayerNorm([num_hidden * 7, width, width])
            nn.LayerNorm([num_hidden * 4, width, width])

        )
    def forward(self, x_t, h_t, c_t):
        combined = torch.cat([x_t, h_t], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,self.num_hidden , dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_t + i * g
        h_next = o * torch.tanh(c_next)
        print("h_new",h_next.size(),c_next.size())

        return h_next, c_next

        '''
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            # nn.LayerNorm([num_hidden * 7, width, width])
            nn.LayerNorm([num_hidden * 4, width, width])

        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        # self.conv_m = nn.Sequential(
        #     nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
        # #     nn.LayerNorm([num_hidden * 3, width, width])
        # )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden , num_hidden, kernel_size=1, stride=1, padding=0)
        # self.conv = nn.Conv2d(in_channels=in_channel + self.hidden_dim,
        #                       out_channels=4 * num_hidden,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=self.bias)
        '''

#LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；https://blog.csdn.net/shanglianlm/article/details/85075706
        '''
    def forward(self, x_t, h_t, c_t):

        print("x_t",x_t.size())
        print("h_t",h_t.size())
        print("c_t",c_t.size())
      
#        print("上一步驟做h和m做conv x也做conv---")
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        # m_concat = self.conv_m(m_t)
        
        print("x_concat",x_concat.size())
        print("h_concat",h_concat.size())

#        print("x分割---")
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        print("i_x",i_x.size())
        print("f_x",f_x.size())
        print("g_x",g_x.size())
#        print("i_x_prime",i_x_prime.size())
#        print("f_x_prime",f_x_prime.size())
#        print("g_x_prime",g_x_prime.size())        
#        print("o_x",o_x.size())        

        
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)#T-1
       
        print("i_h",i_h.size(), "f_h",f_h.size(),"g_h",g_h.size(),"o_h",o_h.size())

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        print("i_t",i_t.size())
        print("f_t",f_t.size())
        print("g_t",g_t.size())        
        print("c_new",c_new.size())  
 

        h_new = (o_x+o_h) * torch.tanh(self.conv_last(c_new))
#        
        print("h_new",h_new.size(),c_new.size())
                # x_t torch.Size([1, 64, 64, 64])
                # h_t torch.Size([1, 64, 64, 64])
                # c_t torch.Size([1, 64, 64, 64])
                # x_concat torch.Size([1, 256, 64, 64])
                # h_concat torch.Size([1, 256, 64, 64])
                # i_x torch.Size([1, 64, 64, 64])
                # f_x torch.Size([1, 64, 64, 64])
                # g_x torch.Size([1, 64, 64, 64])
                # i_h torch.Size([1, 64, 64, 64]) f_h torch.Size([1, 64, 64, 64]) g_h torch.Size([1, 64, 64, 64]) o_h torch.Size([1, 64, 64, 64])
                # i_t torch.Size([1, 64, 64, 64])
                # f_t torch.Size([1, 64, 64, 64])
                # g_t torch.Size([1, 64, 64, 64])
                # c_new torch.Size([1, 64, 64, 64])
                # h_new torch.Size([1, 64, 64, 64]) torch.Size([1, 64, 64, 64])
        return h_new, c_new#, m_new
        '''









