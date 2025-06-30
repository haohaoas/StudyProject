import pandas as pd
import torch
from torch import nn
from  tqdm import tqdm
from EnglishToFrenchOwnTokenizer import Tokenizer
import MyDataset
from model import Seq2Seq,Encoder,Decoder
def main():
    device=torch.device('cuda')
    print('Using device:', device)
    data=MyDataset.get_dataloader('data/eng-fra.csv',batch_size=64,shuffle=True)
    fr_tokenizer_load = Tokenizer.load('model/fr_tokenizer.pkl')
    en_tokenizer_load=Tokenizer.load('model/en_tokenizer.pkl')
    fr_vocab_size=len(fr_tokenizer_load)
    en_vocab_size=len(en_tokenizer_load)
    embed_dim=64
    epochs=100
    hidden_dim=128
    encoder=Encoder(en_vocab_size,embed_dim,hidden_dim)
    decoder=Decoder(fr_vocab_size,embed_dim,hidden_dim)
    model=Seq2Seq(encoder,decoder).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.0001)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss=0
        for src,tgt in tqdm(data):
            src,tgt=src.to(device),tgt.to(device)
            optimizer.zero_grad()
            output=model(src,tgt[:,:-1])#shift for input
            loss=criterion(output.reshape(-1,fr_vocab_size),tgt[:,1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")

if __name__=='__main__':
    main()
