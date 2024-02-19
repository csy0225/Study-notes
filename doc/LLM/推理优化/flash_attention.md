# Flash Attention（减少显存占用，提高推理性能）
Flash attention 是一种解决 transformer 中 Attention 计算过程中对于 memory bound 情况处理的一种方法。他的思想是利用芯片的高速缓存进行计算（利用高带宽），替代传统的 HBM 计算，减少 HBM 访问次数。具体做法是将注意力公式中 Softmax 的计算进行了分块处理,解决了 softmax 以及后面 GEMM 的行方向依赖。
+ 知乎：
  + FlashAttention 的速度优化原理是怎样的？ https://www.zhihu.com/question/611236756/answer/3132304304?utm_id=0
  + FlashAttention:加速计算,节省显存, IO感知的精确注意力 https://zhuanlan.zhihu.com/p/639228219?utm_id=0

## Flash Attention 2
FlashAttention2 在 FlashAttention 算法基础上进行了调整，减少了非矩阵乘法运算（non-matmul）的FLOPs。这是因为现代GPU有针对matmul（GEMM）专用的计算单元（如Nvidia GPU上的Tensor Cores），效率很高。以A100 GPU为例，其FP16/BF16矩阵乘法的最大理论吞吐量为312 TFLOPs/s，但FP32非矩阵乘法仅有19.5 TFLOPs/s，即每个no-matmul FLOP比mat-mul FLOP昂贵16倍。为了确保高吞吐量（例如超过最大理论TFLOPs/s的50％），我们希望尽可能将时间花在matmul FLOPs上。
+ 知乎：FlashAttention2详解（性能比FlashAttention提升200%） https://zhuanlan.zhihu.com/p/645376942

## Flash Attention 3 - Flash Decoding
上面提到 FlashAttention 对 batch size 和 query length 进行了并行化加速，Flash-Decoding 在此基础上增加了一个新的并行化维度：keys/values的序列长度。
FlashAttention 计算更新时，需要逐个更新，有依赖。但是 Flash Decoding 通过最后引入 reduce 计算，可以并行更新。虽然增加了计算量，但是提升了性能。
Flash-Decoding，可以推理过程中显著加速attention操作（例如长序列生成速度提高8倍）。其主要思想是最大化并行加载 keys 和 values 的效率，通过重新缩放组合得到正确结果。
+ 知乎：FlashAttenion-V3: Flash Decoding详解 https://zhuanlan.zhihu.com/p/661478232

理解：文中有个评论，说flash-decoding，有点感觉了，至少第一层合并是并行的，第2层合并似乎还是顺序的？计算注意力分成两个步骤，一个是计算 S(Q*K), 一个是计算 O(softmax（Q*K）*V)，计算 S 的时候是并行的，计算更新 O 的时候还是顺序执行的。

## Flash Attention 4 - 工程化实现，不是新算法的提出
主要是针对动态输入（batch size，query length） 做了判断，优化特定场景下的 kernel 实现。
知乎：【FlashAttention-V4，非官方】FlashDecoding++ https://zhuanlan.zhihu.com/p/665595287
