

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# 加载数据集
dataset = load_dataset("opus_books", "de-en", split="train")
# 读取英语和法语的句子对文件
# 返回两个列表：一个是英文句子列表，一个是法文句子列表
# 提取所有德语句子和英语句子
de_sentences = []
en_sentences = []
for item in dataset:
    # item 是一个字典，例如 {'translation': {'de': '...', 'en': '...'}}
    if 'translation' in item and item['translation']['de'] and item['translation']['en']:
        de_sentences.append(item['translation']['de'].strip())
        en_sentences.append(item['translation']['en'].strip())

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
de = tokenizer(de_sentences, truncation=True, padding=True,  return_tensors="pt")
en = tokenizer(en_sentences, truncation=True, padding=True, return_tensors="pt")

# 定义一个自定义数据集类，继承自 PyTorch 的 Dataset
class TranslationDataset(Dataset):

    # 初始化函数，接收两个参数：源语言的编码结果、目标语言的编码结果
    def __init__(self, encoder_inputs, decoder_inputs):
        self.encoder_inputs = encoder_inputs
        # 保存输入编码结果（例如 input_ids、attention_mask）
        self.decoder_inputs = decoder_inputs
        # 你可以写成 self.encoder_inputs = encoder_inputs
        # self.decoder_inputs = decoder_inputs

    # 返回数据集的长度，也就是样本的数量
    def __len__(self):
        return len(self.encoder_inputs["input_ids"])
        # 返回 encoder 输入的数量即可（例如 len(self.encoder_inputs["input_ids"])）

    # 返回第 idx 条样本，组织成一个字典格式
    def __getitem__(self, idx):
        decoder_input_ids = self.decoder_inputs["input_ids"][idx][:-1]
        labels = self.decoder_inputs["input_ids"][idx][1:]
        labels[labels == tokenizer.pad_token_id] = -100  # 忽略 pad 区域的 loss

        return {
            "encoder_input_ids": self.encoder_inputs["input_ids"][idx],
            "encoder_attention_mask": self.encoder_inputs["attention_mask"][idx],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels
        }
        # 1. 获取 encoder 的 input_ids 和 attention_mask，第 idx 条
        # 2. 获取 decoder 的 input_ids，第 idx 条
        # 3. 构造 decoder_input_ids（去掉最后一个 token）
        # 4. 构造 labels（去掉第一个 token）
        # 5. 把 labels 中为 tokenizer.pad_token_id 的位置换成 -100（忽略 loss）
        # 6. 返回一个字典，包含：
        #    {
        #        'encoder_input_ids': ...,
        #        'encoder_attention_mask': ...,
        #        'decoder_input_ids': ...,
        #        'labels': ...
        #    }

# 导入必要模块
from torch.utils.data import DataLoader

# 实例化你刚才定义的 Dataset 类
train_dataset = TranslationDataset(de, en)

# 定义 DataLoader，用于批量加载数据，batch_size 可以先设成 16
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True  # 训练时随机打乱
)