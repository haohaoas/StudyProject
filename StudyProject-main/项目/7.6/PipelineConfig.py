import torch


class Config(object):
    def __init__(self):
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_data_path='./data/train.txt'
        self.dev_data_path='./data/dev.txt'
        self.test_data_path='./data/test.txt'
        self.embedding_dim=128
        self.pos_dim=32
        self.hidden_dim=128
        self.epochs=10
        self.batch_size=32
        self.max_len=70
        self.lr=1e-3
