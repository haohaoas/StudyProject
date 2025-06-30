import pandas as pd
import torch
from EnglishToFrenchOwnTokenizer import Tokenizer

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.df=pd.read_csv(data_path,sep='\t')
    def __getitem__(self, idx):
        src = torch.tensor(Tokenizer.str_to_list(self.df['English_index'][idx]))
        tgt = torch.tensor(Tokenizer.str_to_list(self.df['French_index'][idx]))

        return src,tgt
    def __len__(self):
        return len(self.df)

def get_dataloader(data_path,batch_size=32,shuffle=True):
    return DataLoader(MyDataset(data_path),batch_size=batch_size,shuffle=shuffle)



