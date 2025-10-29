#@title 安装依赖
!pip install torch torchvision torchaudio seqeval pytorch-crf gensim -q
#@title 导入依赖库
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
import torch.optim as optim
import numpy as np
from seqeval.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
import os
import gensim
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

#@title 数据加载
def read_file(filename):
  contents,labels=[],[]
  with open(filename,'r',encoding='utf-8') as f:
    content,label=[],[]
    for line in f:
      line=line.strip()
      if not line:
        if content:
          contents.append(content)
          labels.append(label)
          content,label=[],[]
        continue
      try:
        word,tag=line.split()
        content.append(word)
        label.append(tag)
      except:
        continue
    if content:
      contents.append(content)
      labels.append(label)
  return contents,labels

base_dir="E:\\d2l-zh\\d2l-zh\\pytorch\\Bi-LSTM_CRF\\data\\"
train_path=os.path.join(base_dir,"train_data.data")
test_path=os.path.join(base_dir,"test_data.data")

train_sentences,train_labels=read_file(train_path)
test_sents,test_labels=read_file(test_path)

train_sents,val_sents,train_labels,val_labels=train_test_split(train_sentences,
train_labels,test_size=0.1,random_state=42)
print("训练集样本数：",len(train_sents))
print("验证集样本数：",len(val_sents))
print("测试集样本数：",len(test_sents))



#@title 构建词汇表和标签字典
word2idx={"<PAD>":0,"<UNK>":1}
for sent in train_sents:
  for word in sent:
    if word not in word2idx:
      word2idx[word]=len(word2idx)

label2idx={"<PAD>":0,"<UNK>":1}
for labels in train_labels:
  for label in labels:
    if label not in label2idx:
      label2idx[label]=len(label2idx)
idx2label={idx:label for label,idx in label2idx.items()}
print("词汇表大小:",len(word2idx))
print("标签数:",len(label2idx))



#@title 加载中文预训练词向量
w2v_path='E:\\d2l-zh\\d2l-zh\\pytorch\\Bi-LSTM_CRF\\data\\sgns_clean.char'
w2v_model=gensim.models.KeyedVectors.load_word2vec_format(w2v_path,binary=False)
embedding_dim=w2v_model.vector_size
vocab_size=len(word2idx)
pretrained_emb=np.random.uniform(-0.25,0.25,(vocab_size,embedding_dim)).astype(np.float32)
for word,idx in word2idx.items():
  if word in w2v_model:
    pretrained_emb[idx]=w2v_model[word]
print(len(pretrained_emb))



#@title 定义数据集和模型
class NERDataset(Dataset):
  def __init__(self,sents,labels,word2idx,label2idx,max_len=100):
    self.sents=sents
    self.labels=labels
    self.word2idx=word2idx
    self.label2idx=label2idx
    self.max_len=max_len

  def __len__(self):
    return len(self.sents)
  def __getitem__(self,idx):
    words=self.sents[idx]
    labels=self.labels[idx]
    word_ids=[self.word2idx.get(word,1) for word in words]
    label_ids=[self.label2idx[t] for t in labels]
    if len(word_ids)<self.max_len:
      word_ids+=[0]*(self.max_len-len(word_ids))
      label_ids+=[0]*(self.max_len-len(label_ids))
    else:
      word_ids=word_ids[:self.max_len]
      label_ids=label_ids[:self.max_len]
    return torch.LongTensor(word_ids),torch.LongTensor(label_ids)
class BiLSTM_CRF(nn.Module):
  def __init__(self,vocab_size,tagset_size,embedding_dim=300,hidden_dim=512,dropout=0.3,
        pretrained_emb=None):
    super(BiLSTM_CRF,self).__init__()
    self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
    #padding_idx特殊填充符号对应id
    if pretrained_emb is not None:
      self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
      self.embedding.weight.requires_grad=True
    self.lstm=nn.LSTM(embedding_dim,hidden_dim//2,num_layers=1,bidirectional=True,
              batch_first=True)
    """
    hidden_dim // 2:隐藏层维度（每个方向的）,因为是双向的，前向+后向的 hidden 拼在一起，最终输出维度会是 hidden_dim
    num_layers=1:LSTM 堆叠层数,通常设为 1 或 2
    bidirectional:是否双向,True 表示使用 BiLSTM（前后各一个 LSTM），False 就是普通单向 LSTM
    batch_first:设为 True 时输入形状是 (batch, seq_len, input_dim)；
    否则是 (seq_len, batch, input_dim)
    """
    self.dropout=nn.Dropout(dropout)
    self.hidden2tag=nn.Linear(hidden_dim,tagset_size)
    self.crf=CRF(tagset_size,batch_first=True)
  def forward(self,sentences,tags,mask=None):
    embeds=self.dropout(self.embedding(sentences))
    #[batch_size,seq]->[batch_size,seq,embedding_dim]
    lstm_out,_=self.lstm(embeds)
    """
    lstm_out, (h_n, c_n) = self.lstm(embeds)
    lstm_out:(batch_size, seq_len, hidden_dim)(forward + backward 拼接)
    h_n:(num_layers * num_directions, batch_size, hidden_size)
    每层、每方向的最后隐藏状态
    c_n:同上,每层、每方向的最后 cell 状态
    """
    lstm_out=self.dropout(lstm_out)
    emissions=self.hidden2tag(lstm_out)#(batch_size, seq_len, num_labels)
    loss=-self.crf(emissions,tags,mask=mask,reduction='mean')
    """
    mask 是一个 布尔（bool）矩阵，形状通常为：
    (batch_size, seq_len)
    它用来告诉模型：
    哪些位置是真实的单词，哪些是 padding（填充）。
    """
    return loss
  def predict(self,sentences,mask=None):
    embeds=self.dropout(self.embedding(sentences))
    lstm_out,_=self.lstm(embeds)
    lstm_out=self.dropout(lstm_out)#只要在预测前调用了model.eval()，它就不会随机丢弃神经元
    emissions=self.hidden2tag(lstm_out)
    return self.crf.decode(emissions,mask=mask)

#@title 训练模型
def train_model(model,train_loader,val_loader,optimizer,epochs=15):
  best_f1=0
  best_state=None
  for epoch in range(epochs):
    model.train()
    total_loss=0
    for X,y in train_loader:
      X, y = X.to(device), y.to(device)
      mask=(X!=0)
      loss=model(X,y,mask=mask)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    preds,true_labels=[],[]
    for X,y in val_loader:
      X, y = X.to(device), y.to(device)
      mask=(X!=0)
      pred=model.predict(X,mask=mask)
      for i,seq in enumerate(pred):
        preds.append([idx2label[t] for t in seq])
        """
        那么 mask（即 X != 0）告诉模型哪些 token 有效，
        多数实现（例如 BiLSTM+CRF 的 predict）
        在预测时会自动忽略 padding 部分
        所以pad部分已经被去了
        """
        true_labels.append([idx2label[int(t)] for t in y[i] if t.item()!=0])
    f1=f1_score(true_labels,preds)
    print(classification_report(true_labels,preds,digits=4))
    print(f"Epoch {epoch+1}/{epochs}, Val F1: {f1:.4f}")

    if f1>best_f1:
      best_f1=f1
      best_state=deepcopy(model.state_dict())

  return best_state


train_dataset=NERDataset(train_sents,train_labels,word2idx,label2idx)
val_dataset=NERDataset(val_sents,val_labels,word2idx,label2idx)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=32)

model=BiLSTM_CRF(len(word2idx),len(label2idx),embedding_dim=embedding_dim,
                 hidden_dim=512,pretrained_emb=pretrained_emb)
model.to(device)
optimizer=optim.Adam(model.parameters(),lr=0.001)
best_state=train_model(model,train_loader,val_loader,optimizer,epochs=15)
torch.save(best_state,"best_model.pt")
print(" 最优模型已保存为 best_model.pt")


#@title 单句预测
def test_sentence(model,sentence,word2idx,idx2label,max_len=100):
  model.eval()
  words=list(sentence.strip())
  #list()"你好"->['你'，'好']
  word_ids=[word2idx.get(word,1) for word in words]
  if len(word_ids)<max_len:
    word_ids+=[0]*(max_len-len(word_ids))
  else:
    word_ids=word_ids[:max_len]
    words = words[:max_len]
  X=torch.LongTensor(word_ids).unsqueeze(0).to(device)
  #unsqueeze(0)增加一维batch_size
  mask=(X!=0)
  pred_tags=model.predict(X,mask=mask)[0]
  tags=[idx2label[t] for t in pred_tags[:len(words)]]

  type_map={'per':'person','loc':'location','org':'organization'}
  entities={"person":[],"location":[],"organization":[]}
  current_type,current_entity=None,[]
  for w,t in zip(words,tags):
    if t.startswith('B-'):
      if current_type and current_entity:
        entities[type_map[current_type]].append(''.join(current_entity))
      current_type=t.split('-')[1].lower()
      current_entity=[w]
    elif t.startswith('I-') and current_type==t.split('-')[1].lower():
      current_entity.append(w)
    else:
      if current_type and current_entity:
        entities[type_map[current_type]].append(''.join(current_entity))
      current_type,current_entity=None,[]
  if current_type and current_entity:
    entities[type_map[current_type]].append(''.join(current_entity))


  return entities
example="中华人民共和国国务院总理周恩来在外交部长陈毅、副部长王东的陪同下,连续访问了埃塞俄比亚等非洲10国以及阿尔巴尼亚"
print(test_sentence(model,example,word2idx,idx2label))

#@title 测试集评估
model.load_state_dict(torch.load("best_model.pt",weights_only=True))
model.eval()


test_dataset=NERDataset(test_sents,test_labels,word2idx,label2idx)
test_loader=DataLoader(test_dataset,batch_size=32)

all_preds,all_labels=[],[]

with torch.no_grad():
  for X,y in test_loader:
    X,y=X.to(device),y.to(device)
    mask=(X!=0)
    pred_tags=model.predict(X,mask=mask)
    for i,seq in enumerate(pred_tags):
      all_preds.append([idx2label[w] for w in seq])
      all_labels.append([idx2label[w.item()] for w in y[i] if w.item()!=0])
print(classification_report(all_labels,all_preds,digits=4))





