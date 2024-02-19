# kv cache
LLM 模型推理计算的过程可以分为 prompt 处理和后续输出 token 的自回归计算。前者有大量数据的矩阵乘，是典型的计算密集型处理，而后者随着LLM的执行，会积累越来越多的对话内容，基于历史输出计算得到新的token输出。 
+ 干货分享｜kv 缓存 (kv cache) 知识 https://baijiahao.baidu.com/s?id=1763480207227229622&wfr=spider&for=pc
+ 知乎： 漫谈 KV Cache 优化方法，深度理解 StreamingLLM https://zhuanlan.zhihu.com/p/659770503
+ 昇腾CANN 7.0 黑科技：大模型推理部署技术解密 https://www.bilibili.com/read/cv27588256/?jump_opus=1