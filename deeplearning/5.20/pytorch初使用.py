import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import torch
from torch import nn
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.sans-serif'] = ['SimHei']

# def create_dataset():
#     x, y, coef = make_regression(n_samples=100, #创建样本数
#                                  n_features=1,
#                                  noise=10,
#                                  coef=True,
#                                  bias=14.5,
#                                  random_state=0)
#     x = torch.tensor(x, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)
#     return x, y, coef
# x, y, coef = create_dataset()
# # print("x shape:",  x.shape)
# # print("y shape:",  y.shape)
# # print("真实权重:",  coef)

#线性回归学习
# model=nn.Linear(in_features=1, out_features=1) #线性回归模型
# loss_fn=nn.MSELoss()#损失函数
# optimizer=torch.optim.SGD(model.parameters(), lr=0.01)#使用随机梯度下降
# loss_list=[]
# for epoch in range(100):
#     #  前向传播
#     y_pred=model(x).squeeze() #[100,1]
#     #  计算损失
#     loss=loss_fn(y_pred, y)
#     #  反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     loss_list.append(loss.item())
#     #  打印结果
#     if epoch%10==0:
#         print(f'epoch:{epoch}, loss:{loss.item()}')
# print("学习到的权重：",model.weight.item())
# print("学习到的偏置：",model.bias.item())
# plt.plot(loss_list)
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.title("计算损失函数下降曲线")
# plt.grid(True)
# plt.show()
# with torch.no_grad():
#     y_pred=model(x).squeeze()
# plt.scatter(x.numpy(),y.numpy(),label='真实数据')
# plt.plot(x.numpy(),y_pred.numpy(),label='预测数据')
# plt.legend()
# plt.title('线性回归拟合效果')
# plt.show()

#多层感知机MLP
def create_dataset():
    x=torch.linspace(-3,3,100).reshape(-1,1)
    y = 5 * x**2 + 3 + torch.randn_like(x) * 0.5
    return x,y,None
x,y,z=create_dataset()
y=y.squeeze()
model=nn.Sequential(
    nn.Linear(1,64),
    nn.ReLU(),
    nn.Linear(64,64),
    nn.ReLU(),
    nn.Linear(64,1)
)
loss_fn=nn.MSELoss()#损失函数
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)#使用随机梯度下降
loss_list=[]
for epoch in range(500):
    #  前向传播
    y_pred=model(x).squeeze() #[100,1]
    #  计算损失
    loss=loss_fn(y_pred, y)
    #  反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    #  打印结果
    if epoch%10==0:
        print(f'epoch:{epoch}, loss:{loss.item()}')
plt.plot(loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss curve')
plt.show()
with torch.no_grad():
    y_pred=model(x).squeeze()
plt.scatter(x.numpy(),y.numpy(),label='真实数据')
plt.plot(x.numpy(),y_pred.numpy(),label='预测数据')
plt.legend()
plt.title('线性回归拟合效果')
plt.show()