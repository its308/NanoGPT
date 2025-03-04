#PLS DON'T RUN THIS AGAIN AND AGAIN, IT GIVES LOAD TO THE COMPUTER

import torch
import torch.nn as nn
from torch.nn import functional as F
#scaled could not run on my computer
# hyperparametrs
batch_size=4   # 32 needed for actual scaling the model
block_size=128 # 256 needed
max_iters=5000
eval_interval=500
lr=3e-4
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=128  # actual needed=384
n_head=6
n_layer=6
dropout=0.2
# --------

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf8') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)


# NOW, WE WANT TO TOKENIZE THE INPUT TEXT
# Tokenisation-converting word into a number not vector of numbers (embedding does that)
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s] # encode is a func here ,lambda is any one line func ,here it takes string (s) as input and gives list of numbers represnting that string
decode=lambda l:''.join([itos[i] for i in l])

data=torch.tensor(encode(text),dtype=torch.long)

# splitting the data into train and validation datasets
n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

#since gpus can process the data in parallel therefore forming the batches of data,so these batches (of no. batch_size )gets processed in parallel
def get_batch(split):

    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,)) #it gives us the starting index from which sequences of chars will be selcted , 2nd argument is just for definig shape
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y=get_batch(split)
            logits,loss=model(X,Y)
            losses[k]=loss
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)  # here's what i'm interested in (i want)
        self.key = nn.Linear(n_embd, head_size, bias=False)  # here's what i have
        self.value = nn.Linear(n_embd, head_size, bias=False)  # here's what i will communincate to you if you find me inetresting
        # since tril is not a parameter to model , you need to register as buffer acc. to pytorch
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        wei=q@k.transpose(-2,-1)*C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # MASKING FURTHER WORDS
        wei=F.softmax(wei,-1)
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(num_heads * head_size,n_embd) # it is needed to capture all the aspects covered by different heads and the finally mixing them(Computing them through mlp)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1) #concatinating all attention heads
        out=self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity : simply mlp"""
    #it is needed bcoz without it tokens talk to each other but did'nt have time to think about what they found from other tokens
    # bcoz once the tokens have gathered the data then they need to think upon that individually
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    # first communication b/w the tokens and then computation of the info
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)  #layer normalisation-->prevents softmax to move toward one hot and prevents unstable training as learning could occur from not from just one or two
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        # x=self.sa(x)
        # x=self.ffwd(x)
        # we will do residual connections  because in deep nn,while backprop it can directly reach to input without making its grad to zero
        #we will do here pro token normalisation
        x=x+self.sa(self.ln1(x))  # do layer normalisation before passing input to attention block
        x=x+self.ffwd(self.ln2(x))          # do layer normalisation before passing input to feedforward network
        return x



class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
    self.positional_embedding_table = nn.Embedding(block_size, n_embd)
    # self.sa_heads = MultiHeadAttention(4,n_embd//4) # i.e. 4 heads of 8-dimensional self-attention.
    # self.ffwd=FeedForward(n_embd)

    # self.blocks=nn.Sequential(
    #     Block(n_embd,4),
    #     Block(n_embd, 4),
    #     Block(n_embd, 4),
    #     nn.LayerNorm(n_embd)
    # )

    # for scaling up the model
    self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
    self.lm_head=nn.Linear(n_embd,vocab_size) # it is language model head i.e. to go from tokens to logits we need a linear layer
    # idx and targets are both (B,T) tensor of integers



  def forward(self,idx,targets=None):
    B,T=idx.shape
    token_emb=self.token_embedding_table(idx) # (B,T,C). - B: batch size,T:time,C:n_embd
    pos_emb=self.positional_embedding_table(torch.arange(T,device=device)) #(T,C)
    x=token_emb+pos_emb #(B,T,C)
    # x=self.sa_heads(x)
    # x=self.ffwd(x)
    x=self.blocks(x)
    logits=self.lm_head(x)  #(B,T,vocab_size)
    if targets is None:
      loss=None
    else:
      B,T,C=logits.shape
      logits=logits.view(B*T,C)
      targets=targets.view(B*T)
      loss=F.cross_entropy(logits,targets)
    return logits,loss

  def generate(self,idx,max_token_len):
    for _ in range(max_token_len):
      idx_crop=idx[:,-block_size:]  # because if idx has more T then our pos. emb. table would run out of scope therefore cropping input
      logits,_=self(idx_crop)
      logits=logits[:,-1,:]
      probs=F.softmax(logits,dim=-1)
      idx_next=torch.multinomial(probs,num_samples=1)
      idx=torch.cat((idx,idx_next),dim=1)
    return idx



model=BigramLanguageModel()
m=model.to(device)

optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

for iter in range(max_iters):

    if iter% eval_interval==0:
        losses=estimate_loss()
        print(f"step {iter}: train loss ={losses['train']:.4f} val loss ={losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # setting all gradints to zero
    loss.backward()  # backprop
    optimizer.step()  # parameters update

context=torch.zeros((1,1),dtype=torch.long,device=device)

print(decode(m.generate(context,max_token_len=500)[0].tolist()))






