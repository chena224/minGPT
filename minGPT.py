# 非本人原创，仅为学习过程中的复现
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size=64
block_size=256
max_iters=5000
eval_interval=500
learning_rate=3e-4
device="cuda" if torch.cuda.is_available() else "cpu"
eval_iters=200
n_embd=384
n_head=6
n_layer=6
dropout=0.2

torch.manual_seed(1337)
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt","r",encoding="utf-8")as f:
    text=f.read()
# 建立词汇表
chars=sorted(list(set(text)))
vocab_size=len(chars)
# 编码与解码
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:[itos[i] for i in l]

# 将数据编码成int，然后返回tensor类型,训练集取前90%，验证集后10%
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# 获取批次数据
def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix]) 
    x,y=x.to(device),y.to(device)
    return x,y

# 评估平均损失，eval_iters的平均损失，每次获取随机batch
@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ["train","val"]:
        losses=torch.zeros(len(eval_iters))
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

"""此位置我想简要说一下整体流程；(不要忽略首先我们把所有参数进行了一次高斯分布的参数初始化)！然后我们写好单头注意力，因为gpt加入掩码-inf,利用下三角矩阵(T,T),(维度(B,T,H>>>B,T,T>>>B,T,H,),然后我们基于此去写多头(B,T,H>>B,T,C>>>信息混合)
之后我们写前馈层,在之后我们写残差块，在输入注意力层与前馈层之前我们先归一化，当然我们也在他们之后包括在注意力比率那里都做了随机失活0.2，最后整合，加入整体的嵌入矩阵与上下文窗口的位置编码"""
# 封装，好用self，全局变量
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))

        self.dropout=nn.Dropout(dropout)

    def forward(self,x):

        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        wei=q@k.transpose(-2,-1)*k.shape[-1]**-0.5  
        wei=wei.masked_fill(self.tril[:T,:T]==0,float("-inf"))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)

        v=self.value(x)
        out=wei@v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(head_size*num_heads,n_embd)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd //n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.lnf=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
# 参数初始化
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    def forward(self,idx,targets=None):
        B,T=idx.shape

        tok_emb=self.token_embedding_table(idx) # B,T,C
        pos_emb=self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x=tok_emb+pos_emb # (B,T,C)
        x=self.blocks(x) # (B,T,C)
        x=self.lnf(x) # (B,T,C)
        logits=self.lm_head(x) # (B,T,vocab_size)

        if targets is  None:
            loss=None
        else:
            B,T,C=logits.shape
            # 此处通道数在第二维，主要因为pytroch要求问题
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-block_size:]
            logits,loss=self(idx_cond)
            logits=logits[:,-1,:] # (B,C)
            probs=F.softmax(logits,dim=-1) # (B,C)
            idx_next=torch.multinomial(probs,num_samples=1) # (B,1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx
    
model=GPTLanguageModel()
m=model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6,"M parameters")

optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    if iter %eval_interval==0 or iter==max_iters-1:
        losses=estimate_loss()
        print(f'step{iter}:train loss{losses["train"]:.4f},val loss {losses['val']:.4f}')

    xb,yb=get_batch("train")

    logits,loss=model(xb,yb)
    # 小tip,直接设置为None,节省内存
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# 
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))






    
    








    
        










                             







    










