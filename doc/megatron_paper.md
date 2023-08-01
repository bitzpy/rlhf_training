# Megatron论文阅读

## Tranformer结构

- <img src="../img/megatron_paper/transformer_arch.png" style="zoom:100%;" align="center" />     

<br>

## GEMM+GELU的并行

### 方案一   

- X = [X1, X2], A = [A1,A2]（X是按列切割， A按行切割）
- Y = GeLU(X1A1+X2A2)    
- GeLU(X1A1+X2A2]) != GeLU(X1A1)+GeLU(X2A2) (因为非线性)    

### 方案二

- A = [A1, A2]  （按列切割）

- [Y1,Y2] = [GeLU(X1A1),GeLU(X2A2)]   

<img src="../img/megatron_paper/mlp_megatron.png" style="zoom100%;" align="left" />     

**论文中给出的切割方式，A矩阵列切割，B矩阵行切割**   

## Self-Attention的并行   

<img src="../img/megatron_paper/self_attention_megatron.png" style="zoom100%;" align="left" />    

**Attention 每个头的计算在一个 gpu 上**



### 参考资料

https://zhuanlan.zhihu.com/p/622212228