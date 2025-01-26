import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap
import random
import pickle




blocksize = 64
batchsize = 128
max_iter = 3000
learning_rate = 3e-4
eval_iter = 100
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

chars = ""
with open('vocab.txt','r', encoding='utf-8') as f:
    text=f.read()
    chars=sorted(list(set(text)))


vocab_size=len(chars)

#tokenizers

char_to_int={ch:i for i,ch in enumerate(chars)}
int_to_char={i:ch for i,ch in enumerate(chars)}

encode= lambda s:[char_to_int[c] for c in s]
decode= lambda i:[int_to_char[c] for c in i]




class Head(nn.Module):
    #it's one head of self attention

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(blocksize, blocksize)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # input will of size Batch, Time-step, channels
        #output will of size Batch, Time-step, head size

        B,T,C =x.shape

        k =self.key(x) #(B,T,headsize)
        q= self.query(x) #(B,T,headsize)

        #complete attention scores ("affinities")

        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0 , float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei =self.dropout(wei)

        #perform the weighted aggregation of the values
        v = self.value(x)

        out = wei @ v

        return out 

class MultiHeadAttention(nn.Module):

    #multiple heads of self-attention in parallel
    
    def __init__(self, num_head, head_size ):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)]) #4 heads running in parallel
        self.proj = nn.Linear(head_size * num_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        

        return out



class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)




class Block(nn.Module):

    def __init__(self,n_embd, n_head):
        super().__init__()

        head_size= n_embd // n_head #number of features that each head will be capturing in MultiHeadAttention
        self.sa = MultiHeadAttention(n_head, head_size) #sa = self attention
        self.ffwd = FeedForward(n_embd) #ffwd= feed forward
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): #here we have used post-norm :)
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)

        return x



class GPTLanguageModel (nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.postion_embedding_table = nn.Embedding(blocksize, n_embd)
        self.block = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) #final layer normalisation
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0 , std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):

        B,T = index.shape
        
        #idx and targets are both (B,T) tesnor of integers
        tok_emb=self.token_embedding_table(index) #B,T,C
        pos_emb=self.postion_embedding_table(torch.arange(T, device=device))
        x= tok_emb+pos_emb #B,T,C
        x=self.block(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        

        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self,index, max_new_tokens):

        for _ in range(max_new_tokens):

            index_cond = index[:, -blocksize:]

            logits, loss= self.forward(index_cond) # to get the prediction

            logits = logits[:,-1,:] #we have to focus on last time step and it's shape is in form of B,T,C

            probs= F.softmax(logits, dim=-1) # it gives the probability distribution

            index_next=torch.multinomial(probs,num_samples=1) # sample from the distribution

            index = torch.cat((index ,index_next), dim=1)

        return index
    

model = GPTLanguageModel(vocab_size)
print("Loading model parameters.......")
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('Loaded model successfully!')
m = model.to(device)



while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(''.join(generated_chars))