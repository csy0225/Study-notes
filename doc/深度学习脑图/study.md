## 1. 基础知识
### 1.1 layer
#### 1.1.1 norm 相关
+ 1.1.1.1 batch norm

    Batch Normalization 的处理对象是对一批样本, 是对这批样本的同一维度特征做归一化.
+ 1.1.1.2 layer norm

    Layer Normalization 的处理对象是单个样本, Layer Normalization 是对这单个样本的所有维度特征做归一化。
+ 小结: 

    Layer Normalization（LN）和Batch Normalization（BN）在使用场景上有一些区别。LN一般用于NLP任务，而BN一般用于CV任务。 BN、LN可以看作横向和纵向的区别。经过归一化再输入激活函数，得到的值大部分会落入非线性函数的线性区，导数远离导数饱和区，避免了梯度消失，这样来加速训练收敛过程。
    BatchNorm这类归一化技术，目的就是让每一层的分布稳定下来，让后面的层可以在前面层的基础上安心学习知识。参考链接： https://zhuanlan.zhihu.com/p/113233908

#### 1.1.2 池化层
+ 1.1.2.1 作用

1. 特征不变性（feature invariant）
汇合操作使模型更关注是否存在某些特征而不是特征具体的位置
可看作是一种很强的先验，使特征学习包含某种程度自由度，能容忍一些特征微小的位移
2. 特征降维
由于汇合操作的降采样作用，汇合结果中的一个元素对应于原输入数据的一个子区域（sub-region），因此汇合相当于在空间范围内做了维度约减（spatially dimension reduction），从而使模型可以抽取更广范围的特征
同时减小了下一层输入大小，进而减小计算量和参数个数

参考链接：https://zhuanlan.zhihu.com/p/545293528
