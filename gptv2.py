import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from collections import OrderedDict

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# HYPERPARAMETERS
batch_size = 32 # số câu trong 1 batch
block_size = 16 # số token trong 1 câu
max_iters = 5000 # số lần lặp để huấn luyện model
eval_interval = 500 # khoảng thời gian giữa các lần đánh giá trên tập train và val
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # số lần lặp để đánh giá
n_embd = 384
n_head = 6 # số lớp multi-attention
n_layer = 6 # số block
dropout = 0.2

torch.manual_seed(42)


## ÁNH XẠ CÁC TOKEN THÀNH IDS
with open('D:\Code\Python\LLMs\BuildGPT\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
    
# unk_token="[UNK]": UNK là một token đặc biệt, được sử dụng để thay thế các từ không có trong từ vựng (từ chưa được huấn luyện). 
    # Khi một từ không được nhận diện, nó sẽ được ánh xạ thành [UNK].
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

# Huấn luyện tokenizer dựa trên văn bản đầu vào. Tokenizer sẽ học cách ánh xạ từng từ trong văn bản thành các token IDs
# add_special_tokens: Đây là các token đặc biệt được thêm vào từ vựng của tokenizer, gồm:
    # [UNK]: Token đại diện cho từ không xác định.
    # [PAD]: Token để padding chuỗi trong các batch có độ dài khác nhau.
    # [CLS]: Token đại diện cho bắt đầu của chuỗi (thường dùng trong các mô hình như BERT).
    # [SEP]: Token đại diện cho dấu phân cách giữa các câu hoặc các đoạn trong chuỗi.
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
tokenizer.add_special_tokens(special_tokens)

# Huấn luyện tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # chia từ dựa trên khoảng trắng
trainer = trainers.WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"])
tokenizer.train_from_iterator([text], trainer)



# encode
output = tokenizer.encode(text)
# print(output.tokens) # list tokens mà tokenizer học được
# print(output.ids) # list ánh xạ từ tokens đến ids



# Trả về kích thước từ vựng, tức là tổng số token (khác nhau) mà tokenizer đã học được sau quá trình huấn luyện
vocab_size = tokenizer.get_vocab_size()
# print(vocab_size)

## TRAIN TEST SPLIT
data = torch.tensor(output.ids, dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # (SQ, SQ)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, seq_len, n_embd) 
        # seq_len: Độ dài của chuỗi (ví dụ số lượng token trong câu)
        # output of size (batch, time-step, head_size)
        B, SQ, NE = x.shape
        q = self.query(x) # (B, SQ, h_s)
        k = self.key(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) # (B, SQ, h_s) @ (B, h_s, SQ) -> (B, SQ, SQ)
        wei = wei / k.shape[-1] ** (0.5) # q.kT / sqrt(len(k))
        wei = wei.masked_fill(self.tril[:SQ, :SQ] == 0, float('-inf')) # (B, SQ, SQ) => thay thế các vị trí = 0 bằng -ìnf
        wei = F.softmax(wei, dim=-1) # (B, SQ, SQ)
        wei = self.dropout(wei)
        
        out = wei @ v # (B, SQ, SQ) @ (B, SQ, h_s) -> (B, SQ, h_s)
        return out
    

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiheadAttention, self).__init__()
        
        self.heads = nn.ModuleList([SelfAttention(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (batch_size, seq_len, head_size * num_heads)
        out = self.dropout(self.proj(out)) # (batch_size, seq_len, head_size * num_heads) => (batch_size, seq_len, n_embd)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super(FeedForward, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout()
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super(Block, self).__init__()
        
        head_size = n_embd // n_head
        self.mt = MultiheadAttention(num_heads=n_head, head_size=head_size)
        self.fw = FeedForward(n_embd=n_embd)
        
        # Chuẩn hóa dữ liệu đầu vào trước khi đưa vào attention và feed-forward
        # => Layer Normalization giúp ổn định và tăng tốc độ huấn luyện bằng cách chuẩn hóa đầu vào trên mỗi batch
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.mt(self.ln1(x)) # resnet
        x = x + self.fw(self.ln2(x))
        return x


class BasicGPT(nn.Module):
    def __init__(self):
        super(BasicGPT, self).__init__()
        
        self.tokenizer_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embbeding_table = nn.Embedding(block_size, n_embd)
        self.ln_emb = nn.LayerNorm(n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        
        # chuyển đổi từ vector embedding sang không gian từ vựng (vocabulary space)
        # final linear
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self.initweight)
    
    def initweight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, idx, targets=None):
        # idx, tagert (B,SQ)
        B, SQ = idx.shape
        token_eb = self.tokenizer_embedding_table(idx) # (B, SQ, n_embd)
        pos_eb = self.position_embbeding_table(torch.arange(0, SQ))  # (SQ, n_embd)
        x = token_eb + pos_eb # (B, SQ, n_embd)
        x = self.ln_emb(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B, SQ, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, SQ, VS = logits.shape 
            logits = logits.view(B*SQ, VS) # flatten data (B, SQ, vocab_size) -> (B*SQ, vocab_size)
            targets = targets.view(B*SQ)
            
            # Hàm cross_entropy tự động tính softmax và chọn token dựa trên phân phối xác suất này 
            loss = F.cross_entropy(logits, targets)
            
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # chỉ lấy block_size từ cuối cùng để dự đoán từ tiếp theo
            
            # get predicts ( trong quá trình generate target = None => shape của logits vẫn là 3 chiều)
            logits, _ = self.forward(idx_cond)
            
            # chỉ lấy token cuối để dự đoán token tiếp theo
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            # Chọn chỉ số tiếp theo từ phân phối xác suất
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Gắn token mới vào câu
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
model = BasicGPT().to(device=device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


def estimate_loss():
    with torch.no_grad():
        out = {}
        model.eval()
        for split in ['train', 'val']: # duyệt qua hai giá trị split là 'train' và 'val'
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
        
        

# create a Pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    
    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    logits, loss = model(xb, yb) # gọi đến forward của BasicGPT
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), "LLMs/BuildGPT/gpt_modelv2.pth")

string = "hello how are you today?"
output_str  = tokenizer.encode(string)
input_encoded = torch.tensor(output_str.ids, dtype=torch.long).unsqueeze(0).to(device)

print(tokenizer.decode(model.generate(input_encoded, max_new_tokens=10)[0].tolist(), skip_special_tokens=True))