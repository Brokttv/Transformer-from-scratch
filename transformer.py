import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Embedding(nn.Module):

  def __init__(self, emb_size :int, vocab_size:int):
    super().__init__()
    self.emb_size = emb_size
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, emb_size)

  def forward(self,x): #x: (batch , seq_len)
    return self.embedding(x) * math.sqrt(self.emb_size) #output(batch,seq_len,emb_size)

    #---> this basically takes a vector of size vocab_size with token ids and matches each id to its corresponding emebedding vector i.e, the corresponding row. each row index is represntative of the token of same idx, e.g, token 34 embedding vector is the row number 34 in the matrix.
    #--->  To emb a sequence, you have to look at each token and retrieve its embedding vector from  the (vocab_size, emb_size) matrix and that will have a shape of (batch_size, seq_len, emb_size)
    #---> nn.Linear without a bias is mathematically equavilant but computationally expensive.


class PositionalEncoding(nn.Module):
  def __init__(self,seq_len, emb_size,dropout):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    pos = torch.arange(seq_len).unsqueeze(1) # (seq_len)
    pe = torch.zeros(seq_len, emb_size)
    div= torch.exp(torch.arange(0,emb_size,2)*-(math.log(10000.0)/emb_size))                                                        
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    pe = pe.unsqueeze(0)   # (1,seq_len, emb_size)
    self.register_buffer("pe",pe)  #---> it registers "pe" in state_dict() as a buffer so when you call model.cuda()/model.to(device), it will be moved to the target device with the learnable parameters without it itself being a learnable param that gets updated by the optimizer.

  def forward(self,x):
    x = x + self.pe[:, :x.size(1), :] # here we slice because, seq_len of embeddings from Embedding Class may not match the seq_len of "pe" so we make sure pe is applied to the input's seq_len. To that extent, it is crucial that you set the "pe" seq_len in Postional Enconding Class to the maximum you'll ever use so that "pe" seq_len >= x seq_len. If it's the other way around, then only "pe" seq_len tokens will have pe and the rest will be ignored whcih will lead to discrepancies in the model's learning.  
    return self.dropout(x)
    


#Learneable PE:
'''def __init__(self, emb_size:int, seq_len:int):
    super().__init__()

    self.pos_tensor = torch.zeros(seq_len, emb_size)
    self.pos = nn.Parameter(self.pos_tensor, requires_grad=True)

  def forward(self,x):
    return self.pos + x'''


class LayerNorm(nn.Module):

  def __init__(self,eps:float, emb_size):
    super().__init__()
    self.eps =eps
    self.gamma = nn.Parameter(torch.ones(emb_size), requires_grad=True)
    self.beta = nn.Parameter(torch.zeros(emb_size), requires_grad=True)

  def forward(self,x):
    mean = x.mean(dim=-1, keepdim= True)
    var = x.var(dim =-1, unbiased=False, keepdim=True)

    return self.gamma * (x - mean )/ torch.sqrt(var + self.eps) + self.beta


class FFN(nn.Module):

  def __init__(self,emb_size, ffn_size, dropout:float):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    self.mlp = nn.Sequential(nn.Linear(emb_size, ffn_size),
                        nn.ReLU(),
                        nn.Linear(ffn_size, emb_size))


  def forward(self,x):
    return self.dropout(self.mlp(x))


class MultiHeadAttention(nn.Module):

  def __init__(self,emb_size,num_heads, cross_attn =False ):
    super().__init__()

    self.num_heads = num_heads
    self.cross_attn = cross_attn
    self.q = nn.Linear(emb_size, emb_size, bias=False)
    self.k = nn.Linear(emb_size, emb_size, bias=False)
    self.v = nn.Linear(emb_size, emb_size, bias=False)
    self.out = nn.Linear(emb_size, emb_size, bias=False)

    assert  emb_size % num_heads == 0, "emb_size must be divisable by num_heads "
    self.head_dim = emb_size // num_heads


  def  Attention(self,q,k,v, mask): #in self-attention q_len=k_len

    att_w = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim) # (batch, num_heads, q_len, k_len)

    if mask is not None:
      # exmaple of causal masking: mask = torch.triu(torch.ones(att_w.shape[-2],att_w.shape[-1]), diagonal= 1).bool()
      if mask.dim() == 2: #(q_len,k_len)
        mask = mask.unsqueeze(0).unsqueeze(0)

      elif mask.dim()==3: #(batch,q_len,k_len)
         mask =  mask.unsqueeze(1)

      att_w = att_w.masked_fill(mask, float('-inf'))

    att_output = F.softmax(att_w,dim=-1) @ v # (batch, num_heads, q_len, head_dim)

    return att_output


  def forward(self,x,y=None,mask=None): # x:decoder self-attention output (query) & y: enoder outputs (key and value)

   if y is None:
    y=x

   if self.cross_attn:
    query = self.q(x)
    key = self.k(y)
    value = self.v(y)


   else:
    query = self.q(x) #(batch,q_len,emb_size)
    key = self.k(x) #(batch,k_len,emb_size)
    value =self.v(x) #(batch,k_len,emb_size)

   # you can use .view() but .reshape() is more versatile as it handles both contiguous and non-contiguous tensors.
   # PS: contigous means that the new shape is compatible by the flat data buffer stride in memory so it can "reinterprete" the data from a different view with no problems.


   query= query.reshape(query.shape[0], self.num_heads, query.shape[-2],self.head_dim) #(batch, num_heads, q_len, head_dim)
   key= key.reshape(key.shape[0], self.num_heads, key.shape[-2],self.head_dim) #(batch, num_heads, k_len, head_dim)
   value= value.reshape(value.shape[0], self.num_heads, value.shape[-2],self.head_dim) #(batch, num_heads, k_len, head_dim)

   attention = self.Attention(query,key,value,mask ) # (batch, num_heads, seq_len, head_dim)

   concat = attention.transpose(1,2).flatten(-2,-1) #(batch, seq_len, emb_size)
   return  self.out(concat) #--> w_out



class SkipConnect(nn.Module):
  def __init__(self,dropout:float, eps:float, emb_size):
    super().__init__()
    self.norm = LayerNorm(eps, emb_size)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x, sublayer,*extra_args):
    return x + self.dropout(sublayer(self.norm(x), *extra_args))
    #PS: you may notice that we apply normalization first to the initial input then wrap in sublayer(e.g.FFN, ATTENTION) as opposed to the original paper. that is simply beacuse applying norm first prevents vanishing grads whwn sublayers are deep as opposed to the 2017 paper


class EncoderBlock(nn.Module):
  def __init__(self,dropout,eps,emb_size,num_heads,ffn_size ):
    super().__init__()

    self.Attention =  MultiHeadAttention(emb_size,num_heads,cross_attn=False)
    self.feed_forward = FFN(emb_size,ffn_size, dropout)
    self.skip  = nn.ModuleList([SkipConnect(dropout, eps, emb_size) for _ in range(2)])

  def forward(self,x, mask=None):
    x = self.skip[0](x, lambda x,m:self.Attention(x,x,m),mask)
    x = self.skip[1](x,lambda x: self.feed_forward(x))

    return x


class Encoder_layers(nn.Module):
    def __init__(self, eps, emb_size, n, num_heads, ffn_size, dropout):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                dropout,
                eps,
                emb_size,
                
                num_heads,
                ffn_size)
            for _ in range(n)
        ])

        self.norm = LayerNorm(eps, emb_size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
  def __init__(self,dropout,eps,emb_size,num_heads,ffn_size):
    super().__init__()

    self.cross_attn = MultiHeadAttention(emb_size,num_heads,True)
    self.masked_attn = MultiHeadAttention(emb_size,num_heads,False)
    self.FFN = FFN(emb_size, ffn_size, dropout)
    self.skip = nn.ModuleList([SkipConnect(dropout, eps,emb_size) for _ in range(3)])


  def forward(self,x,encoder_out,mask, src_mask=None):
    x= self.skip[0](x, lambda x,m:self.masked_attn(x,x,m), mask)
    x = self.skip[1](x, lambda x,enc_out,m: self.cross_attn(x,enc_out,m), encoder_out,src_mask)
    x = self.skip[2](x, lambda x: self.FFN(x))
    return x


class Decoder_layers(nn.Module):
  def __init__(self,dropout,eps,emb_size,num_heads,ffn_size,n):
    super().__init__()

    self.emb_size = emb_size


    self.layers = nn.ModuleList([DecoderBlock(dropout,eps,emb_size,num_heads,ffn_size) for _ in range(n)])
    self.norm = LayerNorm(eps, emb_size)



  def forward(self,x, encoder_out,mask,src_mask=None):
    for layer in self.layers:
      x = layer(x,encoder_out,mask,src_mask)
    x = self.norm(x)

    return x

class linear_proj(nn.Module):
  def __init__(self, emb_size,vocab_size):
    super().__init__()
    self.proj = nn.Linear(emb_size,vocab_size,bias=False)
  def forward(self,x):
    return self.proj(x)



class Transformer(nn.Module):
  def __init__(self,emb_size, trgt_vocab_size,src_vocab_size, trgt_len, src_len, eps, src_n, trgt_n,num_heads,ffn_size, dropout):
    super().__init__()

    self.encoder_embed = Embedding(emb_size,src_vocab_size)
    self.encoder_pos = PositionalEncoding(src_len, emb_size,dropout)
    self.decoder_embed = Embedding(emb_size, trgt_vocab_size)
    self.decoder_pos = PositionalEncoding(trgt_len, emb_size,dropout)
    self.encoder_layers = Encoder_layers(eps, emb_size, src_n,num_heads,ffn_size,dropout)
    self.decoder_layers = Decoder_layers(dropout,eps,emb_size,num_heads,ffn_size,trgt_n)
    self.linear_proj = linear_proj(emb_size, trgt_vocab_size)

  def encoder(self,src_x,mask=None):
    src_x = self.encoder_embed( src_x)
    src_x= self.encoder_pos( src_x)
    src_x = self.encoder_layers( src_x, mask)

    return src_x



  def decoder(self,trgt_x,encoder_out,mask,src_mask=None):
    trgt_x = self.decoder_embed(trgt_x)
    trgt_x = self.decoder_pos(trgt_x)
    trgt_x = self.decoder_layers(trgt_x,encoder_out,mask,src_mask)

    return trgt_x


  def proj(self,decoder_out):
    return self.linear_proj(decoder_out)


  def forward(self,src,trgt,mask,src_mask=None):
    src = self.encoder(src,src_mask)
    trgt = self.decoder(trgt,src,mask,src_mask)
    logits = self.proj(trgt)

    return logits

