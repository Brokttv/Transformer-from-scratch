# Transformer-from-scratch
elaborate transformer implementation + detailed explanation 


# **1. Embeddings:**

Takes the tokens and projects them into a high dimensioinal space of d dimensions (e.g., 512). Imagine a data point represented by one-hot vector of size (vocab_size) and you map it to a 512 dimensional space using a linear map which is a matrix of size (d_model,vocab_size). Now imagine a dense space with d_model hyperplanes (axis) and each data point lies there with say 512 coordinates; each representing a unique feature. Those features evolve to capture different semantics of the one token they represent and that is why they're called learneable parameters; they go through the optimzer and get updated multiple times until they achieve accurate representations.
Take a Dog and a Cat: Initially, both appear similar to the model but then as training progresses, they might be indenitcal along a certain axis (they're both pets) and different along others (not the same animal).

TL;DR: Embeddings allow the model to capture and learn as many information as psooible about each token which helps with generalization.


```
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

```




# **2. Positional Encodings:**

They encode positions of tokens in their respective sequences, and you can do it two ways: Learnable params, so it's just a tensor initialzed with zeros or random values and you add it to the embeddings and the model just learns them iteratively through backprop until they have real values that encode token's postions. On the other hand, you have fixed Sinusoidal encodings as noted in the infamous paper "Attention is All you need" and those are the real deal.

Sinusoidal encodings constitute of (sin,cos) pairs for the following reasons:



*  First of all, sin is assigned to even dims indices and cos to odd ones. This is not intrisic to either sin or cos at all as you can swap them and the math will remain the same. All we care about are the pairs.



*   The pairs unlock the linear relative position property: you can compute relative psotions PE(pos+k) = A . PE(pos) all thanks to the sin(pos+k) and cos(pos+k) formulas. `sin(pos + k) = sin(pos)·cos(k) + cos(pos)·sin(k)` &
 `cos(pos + k) = cos(pos)·cos(k) - sin(pos)·sin(k)`
*   A is a rotation matrix and it linearly transforms a vector of size d_model // 2 of postion "pos" to psoiton "pos+k". Imagine moving couple of angles on a circle from initial one.

*  For each postion "P", those pairs which are a total of d_model // 2 (e.g.,256 if d_model=512) compose different frequencies at different dimensions i.

*   The frequencies inform us about the realtive postion between two tokens. For instance, a token at psotion p and another one at position p+k, if you look at their initial frequencies (the very first pairs values) which are very high, you'll realize that thy're very different signaling that those tokens are close to each other. In that case, their last frequencies will be roughly the same.
*  In contrary, if two tokens are far apart, then their initial frequencies should be ~similar and the last ones very different.




On a side note, we don't take advantage of the linear relative position property computationally as we mostly just pre-compute encodings for a maximum sequence length and use up to the input length of those encoding rows through slicing, then add them to the input embeddings.



```
class PositionalEncoding(nn.Module):
  def __init__(self,max_len, emb_size,dropout):
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

```


# **3. Attention Mechanism:**
Attention mechanism is one of those things that just happen to work in ways nobody has planned and still is subject to studying through reverse engineering so, a solid high-level understanding coupled with a good grasp of code is in my opinion more than enough.

***3.1.Multi-Head Attention:***

For a single head attention, queries,keys and values would be interacting within one dimensonal space and that is d_model of tokens' embeddings. `Q-K-V` are matrices of same shape as embeddings `(batch,seq_len,d_model)` with learnable weights. we initialize each randomly to transform the same input (embeddings) in different ways.

*   Q: how much should I attend to each token
*   K: how can I present myself to the query so it can evaluate my relevance 
*   V: what info should be passed forward.

The Q and K matrices are multiplied by each other to produce attention scores, then the output of this dot ptoduct is mutiplied by V matrix giving each token access to other for enhanced contextual awarness.

Before we multiply attention scores by V, we have to go through two steps: **masking** when needed and **Softmax**.

*   Softmax: Converts raw scores into probabilities summing to 1, 
  emphasizing large scores exponentially while considering all tokens. 
  The `/√d_k` scaling enables smooth gradients.
*   Masking: For each query,we mask the tokens after it so it can only attend to whatever its before it preventing "cheating" and model will be able to predict the next token based on semantics learned from previous ones. Masking is only applied to the decoder self-attention, it's easy to figure out why so I'll leave it to you to ponder.

A single head attention learns limited information about tokens interaction because for each (query,key) pair there is only one attention score because the query only has a 100% attention to allocate once so it compromises by prioritizing certain aspects to base relavance upon. 

On the other hand, multi-head attention has mutliple heads running in parallel with each head operating in a unique subspace of dimension `head_dim = d_model/num_heads` with d_model being divisable by num_heads. In that case, each (quey,key) pairs has "num_heads" different scores, so in every single head, queries choose a different "aspect" to evaluate keys' relevance. Mathematically, that is not guarenteed but still, having a variety of representations that the loss can eveluate and opt for the "perfect" one is the most important gain and almost always works better 
for training.

At the very end, after you split data into heads, you have to bring them back together and mix them forming a fusion of different knowledge the model learned in each subspace. And that is the role of the `w_out` matrix, also initialized like the rest (Q-K-V), it's mutliplied by the concatenation of all heads' outputs.


***3.2.Coss-Attention:***

When using an endocder-decoder transformer, we always have a source `src` and a target `trgt`. So, the source is learned during the encoder phase, the target is learned through self-attention in the deocder then the cross-attention part applies attention mechanism to the target serving as the query matrix and the source serving as the key and value matrix.

Keep in mind that the target and source can be similar or different depending on the task. In translation tasks, cross-attention uses Q from the target (decoder) and K=V from the source (encoder). In decoder-only models like GPT, there's no cross-attention, only self-attention where Q=K=V all come from the same sequence.

***CODE:***

```
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

   concat = attention.transpose(1,2).flatten(-2,-1)
   return  self.out(concat) #--> w_out

```


***3.3.Feed-Forward network:***

An FFN is an MLP layer with a non-linear function that takes the output of the attention of shape `(batch, seq_len, d_model)`, expands it to a higher dimensional space typically 4xd_model and projects it back to its inital space `(d_model)`. It mainly allows each token to reflect on its contextualized representation and refine it.

***CODE:***

```
class FFN(nn.Module):

  def __init__(self,emb_size, ffn_size, dropout:float):
    super().__init__()

    self.dropout = nn.Dropout(dropout)

    self.mlp = nn.Sequential(nn.Linear(emb_size, ffn_size),
                        nn.ReLU(),
                        nn.Linear(ffn_size, emb_size))


  def forward(self,x):
    return self.dropout(self.mlp(x))


```

***3.4. LayerNorm and SkipConnection:***

Lastly, for [LayerNorm ](https://medium.com/towards-artificial-intelligence/initialization-batchnorm-and-layernorm-beyond-textbook-definitions-9306b02c7e9a) and [SkipConnection](https://youtu.be/Q1JCrG1bJ-A?si=C2St_4zaoNYFkbie), you can click on them for the best sources that break them down.



***CODE:***


```
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
```

```
class SkipConnect(nn.Module):
  def __init__(self,dropout:float, eps:float, emb_size):
    super().__init__()
    self.norm = LayerNorm(eps, emb_size)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x, sublayer,*extra_args):
    return x + self.dropout(sublayer(self.norm(x), *extra_args))

```






