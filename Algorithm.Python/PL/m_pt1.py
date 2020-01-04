import sys, os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

def MyCodeFile():
    return(__file__)

#https://www.guru99.com/pytorch-tutorial.html
#https://nn.readthedocs.io/en/rtd/index.html
class NN_1_1(nn.Module):
   def __init__(self, features):
       super(NN_1, self).__init__()
       #super().__init__
       self.bn1 = nn.BatchNorm1d(num_features=features)
       self.layer_In = torch.nn.Linear(features, features)
       self.layer_h1 = torch.nn.Linear(features, round(features/2)+1)
       self.layer_h2 = torch.nn.Linear(round(features/2+1), 10)
       self.layer_Out = torch.nn.Linear(10, 2)
       self.dropoutRate = 0.01

   def forward(self, x):
       x = self.bn1(x)
       x = F.dropout(x, p=self.dropoutRate, training=self.training)
       x = F.relu(self.layer_In(x))
       x = F.dropout(x, p=self.dropoutRate/2, training=self.training)
       x = F.relu(self.layer_h1(x))
       x = F.relu(self.layer_h2(x))
       x = self.layer_Out(x)
       x = F.softmax(x.float(), dim=1) 
       return x

#https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
class NN_1_(nn.Module):
   def __init__(self, features):
       super(NN_1, self).__init__()
       #super().__init__
       self.bn1 = nn.BatchNorm1d(num_features=features)
       self.layer_In = torch.nn.Linear(features, features)
       self.layer_h1 = torch.nn.Linear(features, round(features/2)+1)
       self.layer_h2 = torch.nn.Linear(round(features/2)+1, 10)
       self.layer_Out = torch.nn.Linear(10, 2)
       self.dropoutRate = 0.01
       self.L_out = {}

   def forward(self, x):
       self.L_out['1_x'] = x
       self.L_out['2_bn1'] = self.L_out['1_x'] #self.bn1(self.L_out['1_x'])
       self.L_out['3_do1'] = F.dropout(self.L_out['2_bn1'], p=self.dropoutRate, training=self.training)
       self.L_out['4_in'] = F.relu(self.layer_In(self.L_out['3_do1']))
       self.L_out['5_do2'] = F.dropout(self.L_out['4_in'] , p=self.dropoutRate/2, training=self.training)
       self.L_out['6_h1'] = F.relu(self.layer_h1(self.L_out['5_do2']))
       self.L_out['7_h2'] = F.relu(self.layer_h2(self.L_out['6_h1']))
       self.L_out['8_out'] = F.relu(self.layer_Out(self.L_out['7_h2']))
       self.L_out['9_out_sm'] = F.softmax(self.L_out['8_out'].float(), dim=1) 
       return self.L_out['9_out_sm']

class NN_1__(nn.Module):
   def __init__(self, features):
       super(NN_1, self).__init__()
       #super().__init__
       self.hiddenSize = round(2*features)
       self.layer_In = torch.nn.Linear(features, self.hiddenSize)
       self.layer_h1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_h2 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_h3 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_h4 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_h5 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_Out = torch.nn.Linear(self.hiddenSize, 2)
       self.dropoutRateIn = 0.5
       self.dropoutRate = 0.25
       self.L_out = {}

   def forward(self, x):
       self.L_out['1_x'] = x
       self.L_out['2_do1'] = F.dropout(self.L_out['1_x'], p=self.dropoutRateIn, training=self.training)
       self.L_out['3_in'] = F.relu(self.layer_In(self.L_out['2_do1']))
       self.L_out['4_do2'] = F.dropout(self.L_out['3_in'], p=self.dropoutRate, training=self.training)
       self.L_out['5_h1'] = F.relu(self.layer_h1(self.L_out['4_do2']))
       self.L_out['6_do3'] = F.dropout(self.L_out['5_h1'], p=self.dropoutRate, training=self.training)
       self.L_out['7_h2'] = F.relu(self.layer_h2(self.L_out['6_do3']))
       self.L_out['8_do4'] = F.dropout(self.L_out['7_h2'], p=self.dropoutRate, training=self.training)
       self.L_out['9_h3'] = F.relu(self.layer_h2(self.L_out['8_do4']))
       self.L_out['10_do5'] = F.dropout(self.L_out['9_h3'], p=self.dropoutRate, training=self.training)
       self.L_out['11_h4'] = F.relu(self.layer_h2(self.L_out['10_do5']))       
       self.L_out['12_do6'] = F.dropout(self.L_out['11_h4'], p=self.dropoutRate, training=self.training)
       self.L_out['13_h5'] = F.relu(self.layer_h2(self.L_out['12_do6']))
       #https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044
       #https://medium.com/@himanshuxd/activation-functions-sigmoid-relu-leaky-relu-and-softmax-basics-for-neural-networks-and-deep-8d9c70eed91e
       self.L_out['14_out'] = torch.sigmoid(self.layer_Out(self.L_out['13_h5'])) #F.relu(self.layer_Out(self.L_out['6_do3'])) #torch.sigmoid(self.layer_Out(self.L_out['6_do3']))
       self.L_out['15_out_sm'] = F.softmax(self.L_out['14_out'].float(), dim=1) #if self.training:
       return self.L_out['14_out']

class NN_1(nn.Module):
   def __init__(self, features, softmaxout=False):
       super(NN_1, self).__init__()
       self.features = features
       self.softmaxout = softmaxout
       self.hiddenSize = round(2.0*features)
       self.bn1 = nn.BatchNorm1d(num_features=features)
       self.layer_In = torch.nn.Linear(features, self.hiddenSize)
       self.layer_h1 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_h2 = torch.nn.Linear(self.hiddenSize, self.hiddenSize)
       self.layer_Out = torch.nn.Linear(self.hiddenSize, 2)
       #https://towardsdatascience.com/simplified-math-behind-dropout-in-deep-learning-6d50f3f47275
       self.dropoutRateIn = 0.2
       self.dropoutRate = 0.5
       self.L_out = {}
              
   def forward(self, x):
       self.L_out['1_x'] = x
       self.L_out['1_bn1'] = self.bn1(self.L_out['1_x'])
       self.L_out['2_do1'] = F.dropout(self.L_out['1_bn1'], p=self.dropoutRateIn, training=self.training)
       #https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks
       self.L_out['3_in'] = F.leaky_relu_(self.layer_In(self.L_out['2_do1']))
       self.L_out['4_do2'] = F.dropout(self.L_out['3_in'] , p=self.dropoutRate, training=self.training)
       self.L_out['5_h1'] = F.leaky_relu_(self.layer_h1(self.L_out['4_do2']))
       #self.L_out['6_h2'] = F.leaky_relu_(self.layer_h2(self.L_out['5_h1']))
       self.L_out['7_out'] = F.leaky_relu_(self.layer_Out(self.L_out['5_h1'])) #self.layer_Out(self.L_out['5_h1']) #F.leaky_relu_(self.layer_Out(self.L_out['5_h1']))
       if self.softmaxout:
           self.L_out['8_out_sm'] = F.softmax(self.L_out['7_out'].float(), dim=1)
           return self.L_out['8_out_sm']
       else:
           return self.L_out['7_out']
   def Predict(self, x):
       prediction = self.forward(x)
       #prediction = prediction.data.numpy()
       prediction =np.argmax(prediction.data.numpy(), axis=1) #armagx returns a tuple
       if len(prediction)==1:
           return prediction[0]
       return prediction