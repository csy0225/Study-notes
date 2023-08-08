时间区分： 图分析阶段、图执行阶段。
架构组成：API、模型解析、算子定义、算子 kernel、backends、pass 优化。
预测流程：创建 config -> 创建 predictor -> predictor Init -> predictor Run
 + 创建 config
 + 创建 predictor
 + predictor Init
   + PrepareScope 
     + 创建全局 scope、设备初始化, 设置 device id;初始化 InitDefaultKernelSignatureMap