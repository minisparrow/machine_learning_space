#------------------------------------------------------------------
#  1. dataset 
#------------------------------------------------------------------
import torch 
import matplotlib.pyplot as plt

from graph_linear_network import *

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()
#------------------------------------------------------------------
#  2. set up the network 
#------------------------------------------------------------------
net = graph_linear_network(n_feature = 1, n_hidden = 10, n_output = 1)
print(net)
#------------------------------------------------------------------
# 3.  train the network
#------------------------------------------------------------------
# optimizer 
# loss function
# backward
# update-step 
optimizer = torch.optim.SGD(net.parameters(),lr = 0.2)
loss_func = torch.nn.MSELoss()

for i in range(100):
    predict = net(x)
    loss = loss_func(predict,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[interation]\t%f,\t[loss]\t %f" %(i, loss))
