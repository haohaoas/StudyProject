from torch import nn
from torch import optim

def test01():
    linear=nn.Linear(5,3)
    nn.init.uniform_(linear.weight)
    print(linear.weight.data)

if __name__ == '__main__':
    test01()