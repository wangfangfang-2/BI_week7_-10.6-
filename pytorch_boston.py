# 使用numpy实现Boston房价预测
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 数据加载
data = load_boston()
X = data['data']
y = data['target']
#print(X_)
#print(len(X_)) # 506
y = y.reshape(-1,1)
print(y)
#数据规范化

ss = MinMaxScaler()
X = ss.fit_transform(X)

#数据集切分
#X = torch.from_numpy(X).type(torch.FloatTensor)
#y = torch.from_numpy(y).type(torch.FloatTensor)


train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)
#X = torch.from_numpy(X).type(torch.FloatTensor)
#y = torch.from_numpy(y).type(torch.FloatTensor)
train_x=torch.from_numpy(train_x).type(torch.FloatTensor)
test_x=torch.from_numpy(test_x).type(torch.FloatTensor)

train_y=torch.from_numpy(train_y).type(torch.FloatTensor)
test_y=torch.from_numpy(test_y).type(torch.FloatTensor)
#构造网络
from torch import nn
model = nn.Sequential(
        nn.Linear(13,10),
        nn.ReLU(),
        nn.Linear(10,1)
)
#定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)#learning rate 一般取0.01

#训练，max_epoch为超参数
max_epoch = 300
iter_loss=[]#加一数组，显示图
for i in range(max_epoch):
    #前向传播
    y_pred= model(train_x)
    #计算loss
    loss= criterion(y_pred,train_y)
    #打印loss
    print(i,loss.item())
    iter_loss.append(loss.item())#每次数组结果放入iter_loss的list
    #清空之前的梯度，这是pytorch中特有的一步
    optimizer.zero_grad()
    #梯度反向传播
    loss.backward()
    #权重调整
    optimizer.step()
#测试
output = model(test_x)
predict_list = output.detach().numpy()
print(predict_list)

#测试
output = model(test_x)
predict_list = output.detach().numpy()
print(predict_list)
# 绘制不同iteration的loss
x=np.arange(max_epoch)
y=np.array(iter_loss)
plt.plot(x,y)
plt.title('Loss Value in all iterations')
plt.xlabel('Iterations')
plt.ylabel('Mean Loss Value')
plt.show()
#查看真是值与预测值的散点图
x = np.arange(test_x.shape[0])
y1 = np.array(predict_list)#预测值
y2 = np.array(test_y)#实际值
line1 = plt.scatter(x,y1,c='red')
line2 = plt.scatter(x,y2,c='yellow')
plt.legend([line1,line2],['predict','real'])
plt.title('Prediction vs Real')
plt.ylabel('Boston House Price')
plt.show()