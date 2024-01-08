# Weight only int8
+ 论文地址
  + https://arxiv.org/pdf/2211.10017.pdf

+ 好文解读
  + [大模型推理][WINT8/4](00)🔥通俗易懂讲解-快速反量化算法 https://zhuanlan.zhihu.com/p/657072856 
  + 神经网络中的量化-简介 https://zhuanlan.zhihu.com/p/419052103

## 读后感
Wegiht only int8 和传统的 post-training quantize 不一样， PTQ 量化会同时对 激活 + 权重 进行量化，也就是 kernel 里面的 GEMM 算法是 INT8 * INT8, 但是 Weight only int8 只量化权重，在 GEMM 运算时，将权重会反量化到 fp16，激活和 Bias 还都保持 fp16 类型，GEMM 运算是 FP16 * FP16, 论文中通过对 fp16 数据类型的分析，使用吞吐高的 SIMD vector 指令进行运算，大大减少了推理时延，解决了推理过程中的 memory bound。