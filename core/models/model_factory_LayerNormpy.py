import os
import torch
import torch.nn as nn
from torch.optim import Adam
# from core.models import predrnn
# from core.models import predrnn_LayerNorm
from core.models import convlstm

import numpy as np
from torch.autograd import Variable
class MyMSELoss(torch.nn.Module):
  def __init__(self, weight):
    super(MyMSELoss, self).__init__()
    self.weight =  weight
   
  def forward(self, output, label):
    print("==================")
    # error = output - label#!
    label=label.float() 
    error = label - output#!
    print(output - label)
    '''
    依照label的dBz區間 給予誤差不同權重
    '''
    '''
    error_weight = torch.where((label > 45) & (label <= 65), error*self.weight[0], error)

    error_weight = torch.where((label > 40) & (label <= 45), error*self.weight[1], error_weight)

    error_weight = torch.where((label > 35) & (label <= 40), error*self.weight[2], error_weight)
    error_weight = torch.where((label > 30)& (label <= 35), error_weight*self.weight[3], error_weight)   
    # error_weight = torch.where((label > 25) &(label <= 30), error_weight*self.weight[4], error_weight)
    error_weight = torch.where((label > 0)& (label <= 1), error_weight*self.weight[4], error_weight)   
    '''
    error_weight = torch.where((label < 22), torch.pow(error,2)*self.weight[0], error)

    error_weight = torch.where((label >= 22) & (label < 28), torch.pow(error,2)*self.weight[1], error_weight)

    error_weight = torch.where((label >= 28) & (label < 33), torch.pow(error,2)*self.weight[2], error_weight)
    error_weight = torch.where((label >= 33) & (label < 40), torch.pow(error,2)*self.weight[3], error_weight)   
    # error_weight = torch.where((label > 25) &(label <= 30), error_weight*self.weight[4], error_weight)
    error_weight = torch.where((label >= 40) & (label < 45), torch.pow(error,2)*self.weight[4], error_weight) 

    error_weight = torch.where((label >= 45), torch.pow(error,2)*self.weight[5], error_weight) 
    # error_weight =error_weight.half()#!

        # self.weight = [1,2,5,10,30]#!

# 1, x < 2
# 2, 2 ≤ x < 5
# 5, 5 ≤ x < 10
# 10, 10 ≤ x < 30
# 30, x ≥ 30
# '''
# print("2 rainfall_to_dBZ=",rainfall_to_dBZ(2))#22
# print("5 rainfall_to_dBZ=",rainfall_to_dBZ(5))#28
# print("10 rainfall_to_dBZ=",rainfall_to_dBZ(10))#33
# print("30 rainfall_to_dBZ=",rainfall_to_dBZ(30))#40
    print("加權後",error_weight)
    print("加權後 大小",error_weight.size())

    # error_weight = torch.pow(error_weight,2)
    # print("平方",error_weight)
  
    error_weight_mean = torch.mean(error_weight)
    print("avg=",error_weight_mean)
    
    error_weight_mean = torch.sqrt(error_weight_mean)#?
    error_weight_mean =error_weight_mean#.half()#!
    # print("sqrt=",error_weight_mean)

    return error_weight_mean
class Model(object):
    def __init__(self, configs):
        self.configs = configs
        print("self.configs=",self.configs)
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            # 'predrnn': predrnn_LayerNorm.RNN
            'convlstm': convlstm.ConvLSTM

        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            # print("Network=",Network)
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)

            # self.network = Network(self.num_layers, self.num_hidden, width=512,filter_size=5,stride=1,layer_norm=0).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        # self.MSE_criterion = nn.MSELoss()
        self.weight = [1,2,5,10,30,40]#!
      
        self.custom_criterion = MyMSELoss(self.weight)#!
    def save(self, model_name,save_path):
        stats = {}
        stats['net_param'] = self.network.state_dict()
#        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        
#        checkpoint_path = os.path.join(self.configs.save_dir, 'model.pkl'+'-'+str(itr))
#        torch.save(stats, checkpoint_path)
#        model_name = 'mode_haveLayerNorm_2y3m_itr{}'.format(itr)
        save_path = os.path.join(save_path,'{}'.format(model_name))
        torch.save(stats, save_path)
        print("save model to %s" % save_path)

#    def load(self, checkpoint_path):
#        print('load model:', checkpoint_path)
#        stats = torch.load(checkpoint_path)
#        self.network.load_state_dict(stats['net_param'])
    def load(self, save_path, model_name):
        save_path = os.path.join(save_path, model_name)
        print('load model:', save_path)
        stats = torch.load(save_path)
        self.network.load_state_dict(stats['net_param'])
    def train(self, frames, mask):
        print("model ")
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
#        print("train")
        # print("frames_tensor",np.array(frames_tensor).shape)
#        frames_tensor (8, 20, 16, 16, 16)
#        mask_tensor (8, 9, 16, 16, 16)

        # frames_tensor = Variable(frames_tensor, requires_grad=True)

        # with torch.no_grad():
        # if torch.cuda.is_available():
            # a = torch.rand([3,3]).cuda()
            # frames_tensor=frames_tensor.cuda()
        next_frames = self.network(frames_tensor, mask_tensor)
#        print("next_frames",np.array(next_frames).shape)
        
#        frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape) 
#        print("next_frames.dtype",next_frames.dtype)
        
#        next_frames1=next_frames.detach().numpy()
#        print("next_frames1",np.array(next_frames1).shape)   
#        print("frames_tensor[:, 1:].dtype",frames_tensor[:, 1:].dtype)
     
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        loss = self.custom_criterion(next_frames, frames_tensor[:, 1:])#!
        print("use custom_criterion")
        print("loss=",loss)       
# frames_tensor1=frames_tensor[:, 1:].detach().numpy()
#        print("frames_tensor1",np.array(frames_tensor1).shape)   
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        frames_tensor = Variable(frames_tensor, requires_grad=False)

        with torch.no_grad():
            next_frames = self.network(frames_tensor, mask_tensor)
        return next_frames.detach().cpu().numpy()