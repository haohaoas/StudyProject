import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

'''
# 改动需求： 把老的dataset改成符合当前任务的dataset,支持按批填充
# 具体细节：
# 1. dataset中需要传入源和目标语言的tokenizer,直接在__getitem__方法内索引化
# 2. 目前序列前后+<sos>和<eos>
# 3. 在get_dataloader中定义一个collate_fn方法，用以批量加载时调用padding补全。
# 4. 将collate_fn方法传参于Dataloader的collate_fn参数
# 5. 文件路径名之类的变一变
'''

class MyDataset(Dataset):

    def __init__(self,data_path,src_tokenizer,tgt_tokenizer):
        super().__init__()
        self.df = pd.read_csv(data_path, header=None, names=['en', 'fr'])
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. dataset中需要传入源和目标语言的tokenizer,直接在__getitem__方法内索引化
        # 用tokenizer.encode方法索引化
        src = torch.tensor(self.src_tokenizer.encode(self.df['en'][idx]))
        # 2. 目前序列前后+<sos>和<eos>
        tgt = torch.tensor([1]+self.tgt_tokenizer.encode(self.df['fr'][idx])+[2])
        return src,tgt



def get_dataloader(data_path,src_tokenizer,tgt_tokenizer,batch_size=32,shuffle = True):
    # 3. 在get_dataloader中定义一个collate_fn方法，用以批量加载时调用padding补全。
    def collate_fn(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=src_tokenizer.word2idx['<pad>'])
        # 源序列并行输入给transformer，transformer的自主层计算注意力分数时，
        # 并不考虑顺序的，顺序信息是靠位置编码提供的，所以padding在左或右都可以
        tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_tokenizer.word2idx['<pad>'])
        # 目标序列右测填充始终是必要的,简单理解就是不要先预测<pad>
        return src, tgt

    # 4. 将collate_fn方法传参于Dataloader的collate_fn参数
    # 注意：此处dataset需传入tokenizer，所以get_dataloader这个方法也得传入tokenizer
    return DataLoader(MyDataset(data_path,src_tokenizer,tgt_tokenizer),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn = collate_fn)