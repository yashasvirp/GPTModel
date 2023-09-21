import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import mmap
import random
import argparse

parser = argparse.ArgumentParser(description='Something')
parser.add_argument('--batch-size', type=str, required=True, help='Provide batch size')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# --------- Hyperparameters ------------------
blk_size = 32
batch_size = 128 # 32 blocks stacked together
chkpt = 100 # Checkpoint for every 100 iterations
max_iter = 1000   # maximum number of iterations
lr = 3e-4              # Learning rate
n_embed = 384     # dimension of embedding vector
n_EDlayers = 4 # number of encoders and decoders in the a
n_head = 4
dropout = 0.2

# -------------- Data Handling -----------------
char = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    txt = f.read()
    chars = sorted(list(set(txt)))
 
vocab_size = len(chars)

# character-level tokenization

str_to_int = {ch : i for i,ch in enumerate(chars)}
int_to_str = {i : ch for i,ch in enumerate(chars)}
enc = lambda x : [str_to_int[c] for c in x]  # encoding char to int
dec = lambda x : ''.join([int_to_str[i] for i in x])  # decoding int to char

import mmap
import random
from torch.utils.data import DataLoader, TensorDataset


def get_data(split):
    fname = 'output_train.txt' if split == 'train' else 'output_val.txt'
    with open(fname, 'r') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            fsize = len(mm)     # determine file size and random pos to start reading
            start = random.randint(0, fsize - blk_size*batch_size) # from start to just before the end so that lookahead is possible if required
            
            #seek to that pos and start reading the block of text
            mm.seek(start)
            blk = mm.read(blk_size*batch_size-1)
            
            # decode block to a string and ignoring any invalid byte sequences
            dec_blk = blk.decode('utf-8', errors='ignore').replace('\r','')

            # Train and test split
            data = torch.tensor(enc(dec_blk), dtype=torch.long)
    
    return data


def get_batch(part):
    d = get_data(part)
    idx = torch.randint(len(d) - blk_size, (batch_size,))
    x = torch.stack([d[i:i+blk_size] for i in idx])
    y = torch.stack([d[i+1:i+1+blk_size] for i in idx])
    return x.to(device), y.to(device)


# --------------- Initializing Neural net ----------------------------
# Head
class Head(nn.Module):
    ''' self attention head'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(blk_size, blk_size))) # for masking to avoid lookahead
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, T, C = X.shape
        k = self.key(X) # (B, T, head_size)
        q = self.query(X) # # (B, T, head_size)
        # computing attention score
        wght = q @ k.transpose(-2,-1) * k.shape[-1]**0.5 # Flipping last two dim, and scaling by 1/sq.root
        wght = wght.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wght = F.softmax(wght, dim=-1)
        wght = self.dropout(wght)

        # perform weihted augmentation on values
        v = self.value(X)
        return wght @ v


class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(head_size * n_heads, n_embed) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        out = torch.cat([h(X) for h in self.heads], dim=-1) # dim -> (B, T, 4*features)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, X):
        return self.network(X)


class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head   # how many features does each head capture
        self.ma = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.ma(x)
        x = self.ln1(x + y)   # Adding and normalizing
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x    

class GPTLangModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embed)   # token embedding lookup table
        self.pos_embed_table = nn.Embedding(blk_size, n_embed)   # positional embedding table
        self.layers = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_EDlayers)])
        self.l_fin = nn.LayerNorm(n_embed)       # Final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)    # Final linear layer after encoders-decoders
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
    
    def forward(self, idx, targets=None):
        batch, time = idx.shape
        
        # idx and targets are tensors of shape (batch*time)
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed_table(torch.arange(time, device=device))
        x = tok_emb + pos_emb
        x = self.layers(x)
        x = self.l_fin(x) 
        logits = self.lm_head(x)
        
        
        if targets is None:
            loss = None
        else:
            batch, time, channels = logits.shape       # time is a sequential dimension, channels = vocab size (or) number of classes 
            logits = logits.view(batch*time, channels)   # reshaping logits(B,T,C) dimensions to (B*T, C)
            targets = targets.view(batch*time)    
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    
    def generate(self, idx, maxNewTokens):

        for _ in range(maxNewTokens):
            logits, loss = self.forward(idx)
            logits = logits[:, -1, :]    # focusing on last timestep as it is a bigram model (single prev char)
            probs = F.softmax(logits, dim = -1)   # get the probabilities of last dimension
            idxNxt = torch.multinomial(probs, num_samples=1) # sample from those probabilities
            idx = torch.cat((idx, idxNxt), dim=1)
        
        return idx

# ----------------- Reloading the pre-trained model and saving for better usage -----------

# uncomment below ONLY if there is a pre-trained model that you want to load and retrain
'''
model = GPTLangModel(vocab_size)

with open('model_01.pkl','rb') as f:
    model = pickle.load(f)
print('loaded successfully')
m = model.to(device)
'''

# ----------------- loss function --------------
# loss function
@torch.no_grad()

def calc_loss():
    out = {}
    
    model.eval()
    
    for i in ['train', 'val']:
        losses = torch.zeros(chkpt)
        
        for j in range(chkpt):
            X, y = get_batch(i)
            logits, loss = model(X, y)
            losses[j] = loss.item()
        
        out[i] = losses.mean()    
    
    model.train()
    
    return out

# ----------------- Training loop --------------------

# Training loop

optim = torch.optim.AdamW(model.parameters(), lr=lr) # defining optimizer

for i in range(max_iter):
    if i % chkpt == 0:
        losses = calc_loss()
        print(f"Epoch: {i} - Train loss: {losses['train']:.4f}, Validation loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

print(loss.item())

# ---------------- Saving the model -------------------------

import pickle
model = GPTLangModel(vocab_size)

name = str(input('Enter file name to be saved with: ')) + '.pkl'
with open(name,'rb') as f:
    model = pickle.load(f)
print('loaded successfully')
m = model.to(device)
with open('model_01.pkl','wb') as f:
    pickle.dump(model, f)
print('model saved')
