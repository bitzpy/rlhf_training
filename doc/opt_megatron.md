# 用 megatron 的方式改写 opt 模型

## opt 模型的结构

opt 模型结构类似GPT模型的结构，由多个 transformer 解码器（如下图所示）堆叠而成。所以我们主要要解决的是 Mlp 和 Masked Multi Self Attention的模型并行。      

<img src="../img/opt_megatron/transformer_decoder.png" style="zoom:100%;" align="center" />

​          

### Masked Multi Self Attention的并行   

```python
class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))

        if mask is not None:
          attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class Mutihead_Attention(nn.Module):
    def __init__(self):
        super(Mutihead_Attention, self).__init__()
        self.d_k = 64
        self.d_v = 64
        self.d_model = 512
        self.n_heads = 8

        self.w_q = nn.Linear(self.d_model, self.d_k , bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_k , bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_k , bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)

    def forward(self, x, mask):
        #b x lq x dv
        residual = x
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        if mask is not None:
          mask = mask.unsqueeze(1)   # For head axis broadcasting.

        out, attn = self.attention(q, k, v, mask=mask)

        out_tensor_list = [ torch.zeros_like(out).to(device) for i in range(self.n_heads)]
        out_tensor_list [torch.local_rank()] = out
        dist.all_gather(out_tensor_list, out)
        out = torch.cat(out_tensor_list,axis=-1)
        out = out + residual
        out = self.layer_norm(out)
        return out   
```

### MLP的并行  

```python
class TPRowLR(nn.Module):
    def __init__(self):
        super(TPRowLR, self).__init__()
        self.fc = nn.Linear(512, 64)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=1e-6)
    def forward(self, x):
        out = torch.mm(x, self.fc.weight.transpose(0,1))
        out_tensor_list = [ torch.zeros_like(out).to(device) for i in range(8)]
        out_tensor_list [torch.local_rank()] = out
        dist.all_gather(out_tensor_list, out)
        
        out = torch.cat(out_tensor_list,axis=-1)
        out = torch.add(out, self.fc.bias)
        out = self.layer_norm(out)
        return out