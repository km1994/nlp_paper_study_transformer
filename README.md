# 【关于 NLP】 那些你不知道的事——Transformer 篇

> 作者：杨夕
> 
> 介绍：研读顶会论文，复现论文相关代码
> 
> NLP 百面百搭 地址：https://github.com/km1994/NLP-Interview-Notes
> 
> **[手机版NLP百面百搭](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=3&sn=5d8e62993e5ecd4582703684c0d12e44&chksm=1bbff26d2cc87b7bf2504a8a4cafc60919d722b6e9acbcee81a626924d80f53a49301df9bd97&scene=18#wechat_redirect)**
> 
> 推荐系统 百面百搭 地址：https://github.com/km1994/RES-Interview-Notes
> 
> **[手机版推荐系统百面百搭](https://mp.weixin.qq.com/s/b_KBT6rUw09cLGRHV_EUtw)**
> 
> 搜索引擎 百面百搭 地址：https://github.com/km1994/search-engine-Interview-Notes 【编写ing】
> 
> NLP论文学习笔记：https://github.com/km1994/nlp_paper_study
> 
> 推荐系统论文学习笔记：https://github.com/km1994/RS_paper_study
> 
> GCN 论文学习笔记：https://github.com/km1994/GCN_study
> 
> **推广搜 军火库**：https://github.com/km1994/recommendation_advertisement_search
![](other_study/resource/pic/微信截图_20210301212242.png)

> 手机版笔记，可以关注公众号 **【关于NLP那些你不知道的事】** 获取，并加入 【NLP && 推荐学习群】一起学习！！！

> 注：github 网页版 看起来不舒服，可以看 **[手机版NLP论文学习笔记](https://mp.weixin.qq.com/s?__biz=MzAxMTU5Njg4NQ==&mid=100005719&idx=1&sn=14d34d70a7e7cbf9700f804cca5be2d0&chksm=1bbff26d2cc87b7b9d2ed12c8d280cd737e270cd82c8850f7ca2ee44ec8883873ff5e9904e7e&scene=18#wechat_redirect)**

- [【关于 NLP】 那些你不知道的事——Transformer 篇](#关于-nlp-那些你不知道的事transformer-篇)
  - [【关于 transformer 】 那些的你不知道的事](#关于-transformer--那些的你不知道的事)
  - [参考资料](#参考资料)
  

## 【关于 transformer 】 那些的你不知道的事

- [【关于Transformer】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/)  transformer 论文学习
  - [【关于Transformer】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/Transformer/)
    1. 为什么要有 Transformer?
    2. Transformer 作用是什么？
    3. Transformer 整体结构怎么样？
    4. Transformer-encoder 结构怎么样？
    5. Transformer-decoder 结构怎么样?
    6. 传统 attention 是什么?
    7. self-attention 长怎么样?
    8. self-attention 如何解决长距离依赖问题？
    9. self-attention 如何并行化？
    10. multi-head attention 怎么解?
    11. 为什么要 加入 position embedding ？
    12. 为什么要 加入 残差模块？
    13. Layer normalization。Normalization 是什么?
    14. 什么是 Mask？
    15. Transformer 存在问题？
    16. Transformer 怎么 Coding?
  - [【关于 Transformer-XL】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/T3_Transformer_XL/)
    - 动机
      - RNN：主要面临梯度消失或爆炸（gradient vanishing and explosion），解决方法集中在优化方法、初始化策略、辅助记忆单元的研究上。
      - vanilla Transformer：最长建模长度是固定的，无法捕捉更长依赖关系；等长输入序列的获取通常没有遵循句子或语义边界（出于高效考虑，往往就是将文本按长度一段段截取，而没有采用padding机制），可能造成上下文碎片化（context fragmentation）。
    - 方法
      - 引入循环机制（Reccurrence，让上一segment的隐含状态可以传递到下一个segment）：将循环（recurrence）概念引入了深度自注意力网络。不再从头计算每个新segment的隐藏状态，而是复用从之前segments中获得的隐藏状态。被复用的隐藏状态视为当前segment的memory，而当前的segment为segments之间建立了循环连接（recurrent connection）。因此，超长依赖性建模成为了可能，因为信息可以通过循环连接来传播。
      - 提出一种新的相对位置编码方法，避免绝对位置编码在循环机制下的时序错乱：从之前的segment传递信息也可以解决上下文碎片化的问题。更重要的是，本文展示了使用相对位置而不是用绝对位置进行编码的必要性，这样做可以在不造成时间混乱（temporal confusion）的情况下，实现状态的复用。因此，作为额外的技术贡献，文本引入了简单但有效的相对位置编码公式，它可以泛化至比在训练过程中观察到的长度更长的注意力长度。
  - [【关于 SHA_RNN】 那些的你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/SHA_RNN_study/)
    - 论文名称：Single Headed Attention RNN: Stop Thinking With Your Head 单头注意力 RNN: 停止用你的头脑思考
  - [【关于 Universal Transformers】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/T4_Universal_Transformers/)
  - [【关于Style_Transformer】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/Style_Transformer/LCNQA/)
  - [【关于 Linformer 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/ACL2020_Linformer)
    - 论文标题：《Linformer: Self-Attention with Linear Complexity》
    - 来源：ACL 2020
    - 链接：https://arxiv.org/abs/2006.04768
    - 参考：https://zhuanlan.zhihu.com/p/149890569
  - [【关于 Performer 】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_study/Performer) **【推荐阅读】**
    - 阅读理由：Transformer 作者 Krzysztof Choromanski 针对 Transformer 问题的重新思考与改进
    - 动机：Transformer 有着巨大的内存和算力需求，因为它构造了一个注意力矩阵，需求与输入呈平方关系;
    - 思路：使用一个高效的（线性）广义注意力框架（generalized attention framework），允许基于不同相似性度量（核）的一类广泛的注意力机制。
    - 优点：该方法在保持线性空间和时间复杂度的同时准确率也很有保证，也可以应用到独立的 softmax 运算。此外，该方法还可以和可逆层等其他技术进行互操作。
  - [【关于 Efficient Transformers: A Survey】 那些你不知道的事](https://github.com/km1994/nlp_paper_study_transformer/tree/master/DL_algorithm/transformer_survey/Performer)
    - 一、摘要
    - 二、Transformer 介绍
    - 三、Efficient Transformers
      - 3.1 Fixed patterns（FP）
        - 3.1.1 Fixed patterns（FP） 介绍
        - 3.1.2 Fixed patterns（FP） 类别
      - 3.2 Combination of Patterns (CP)
        - 3.2.1 Combination of Patterns (CP) 介绍
        - 3.2.2 Combination of Patterns (CP)  类别
        - 3.2.3 Fixed patterns（FP） vs 多Combination of Patterns (CP)
      - 3.3 Learnable Patterns (LP)
        - 3.3.1 Learnable Patterns (LP) 介绍
        - 3.3.2 Learnable Patterns (LP)  类别
        - 3.3.3 Learnable Patterns (LP)  优点
      - 3.4 Memory
        - 3.4.1 Memory 介绍
        - 3.4.2 Memory 类别
      - 3.5 Low-Rank 方法
        - 3.5.1 Low-Rank 方法 介绍
        - 3.5.2 Low-Rank 方法 类别
      - 3.6 Kernels 方法
        - 3.6.1  Kernels 方法 介绍
        - 3.6.2  Kernels 方法 代表
      - 3.7  Recurrence 方法
        - 3.7.1  Recurrence 方法 介绍
        - 3.7.2  Kernels 方法 代表
    - 四、Transformer 变体 介绍
      - 4.1 引言
      - 4.2 Memory Compressed Transformer 
      - 4.3 Image Transformer 
      - 4.4 Set Transformer 
      - 4.5 Sparse Transformer
      - 4.6 Axial Transformer
      - 4.7 Longformer
      - 4.8  Extended Transformer Construction (ETC)（2020）
      - 4.9  BigBird（2020）
      - 4.10  Routing Transformer
      - 4.11  Reformer（2020）
      - 4.12  Sinkhorn Transformers
      - 4.13  Linformer
      - 4.14   Linear Transformer
      - 4.15  Performer（2020）
      - 4.16  Synthesizer models（2020）
      - 4.17  Transformer-XL（2020）
      - 4.18  Compressive Transformers
    - 五、总结

## 参考资料

1. [【ACL2020放榜!】事件抽取、关系抽取、NER、Few-Shot 相关论文整理](https://www.pianshen.com/article/14251297031/)
2. [第58届国际计算语言学协会会议（ACL 2020）有哪些值得关注的论文？](https://www.zhihu.com/question/385259014)
