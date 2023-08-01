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

## Embedding层的并行

把词表分别存在不同的GPU上，例如第一块GPU查询结果是[e1,0,0,e4]，第二块结果是[0,e2,0,0]，第三块结果是[0,0,e3,0]，然后通过AllReduce 即可得到embedding。

## 交叉熵层的并行

我们输出过完embedding 的结果:[Y1,Y2]=Y，Y1,Y2在不同的GPU上。

我们可以先每个GPU各自算指数求和，然后进行AllReduce得到 softmax 的分母sum(e)，然后每块GPU 可以算各自的 loss，最后在通过AllReduce得到总 loss。

![image-20230801180255352](/Users/zhangpuchang/Library/Application Support/typora-user-images/image-20230801180255352.png)





### 参考资料

https://zhuanlan.zhihu.com/p/622212228

https://www.bilibili.com/video/BV1nB4y1R7Yz/?spm_id_from=333.999.0.0&vd_source=6876da544cecede8a397f4a03ad364df