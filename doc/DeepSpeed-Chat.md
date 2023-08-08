# DeepSpeed-Chat
## 一、 环境准备
```
1. conda 安装
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh
sh Miniconda3-4.7.12-Linux-x86_64.sh

2. 使用 conda 创建 python 3.10环境
conda create -n py3.10 python=3.10

3. torch 安装
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
参考 torch 官网： https://pytorch.org/get-started/previous-versions
```

## 二、 数据集准备
```
git clone https://huggingface.co/datasets/Dahoas/rm-static
git clone https://huggingface.co/datasets/Dahoas/full-hh-rlhf
git clone https://huggingface.co/datasets/Dahoas/synthetic-instruct-gptj-pairwise
git clone https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets
git clone https://huggingface.co/datasets/stanfordnlp/SHP
git clone https://huggingface.co/datasets/openai/webgpt_comparisons
```
## 参考文献
https://zhuanlan.zhihu.com/p/626214655?utm_id=0


## 三、DeepSpeed 训练 -- 一键式RLHF训练 DeepSpeed Chat
### 3.1 SFT 监督微调
#### 3.1.1 概念
+ 监督微调（SFT） 
    + 使用精选的人类回答来微调预训练的语言模型以应对各种查询
+ 标记器 (tokenizer)
    + 标记器是用来对文本进行预处理的一个工具。
    + 第一步：标记器把输入的文档进行分割，将一个句子分成单个的 word，这些句子进行分割以后得到的单个的 word 被称为 tokens。
    + 第二步：将文本中的每个词转成数字,送入到模型当中。其中，为了实现 tokens -> 数字 的转换(文本编码)，这个词汇表在我们进行实例化并指明模型的时候下载的，这个标记器使用的词汇表与模型预训练时使用的词汇表相同。
+ Embedding 简介
    + 参考链接：https://baijiahao.baidu.com/s?id=1765559176667572307&wfr=spider&for=pc
+ 深度学习中 shuffle 的作用
    + 可以防止训练过程中的模型抖动，有利于模型的健壮性。
    + 可以防止过拟合，并且使得模型学到更加正确的特征。
    + 参考链接：https://blog.csdn.net/zhang_j_mail/article/details/128208973
+ 注意力机制中的 attention_mask 详解
    + 参考链接： https://baijiahao.baidu.com/s?id=1771269227526224754&wfr=spider&for=pc
    + attention_mask 在处理多个序列时的作用： https://zhuanlan.zhihu.com/p/414511434
+ 梯度累加(gradient accumulation)
    + 梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加，累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。
    + 一定条件下，batchsize 越大训练效果越好，梯度累加则实现了 batchsize 的变相扩大，如果accumulation_steps 为 8，则batchsize '变相' 扩大了8倍，使用时需要注意，学习率也要适当放大。
    + 参考链接： https://www.zhihu.com/question/303070254/answer/573037166
+ 学习率
    + 参考链接：http://pointborn.com/article/2020/10/6/989.html
+ 神经网络参数更新机制
    + 方法：梯度下降 + 反向传播
    + 将输出误差反向传播给网络参数，以此来拟合样本的输出。本质上是最优化的一个过程，逐步趋向于最优解。
+ Sampler与DistributedSampler(Troch) 区别
    + Distributed sampler将数据划分为num_gpus份，总数据量为1个dataset，而sampler则是每个GPU各自维护一份数据，总数据量为num_gpus个dataset。
+ Weight decay(权重衰减)
    + 一种正则化技术，目的是防止过拟合，减小方差，做法是在训练的 Loss 函数后面增加一个惩罚项，参数越大，训练的 Loss 越大，所以权重衰减的目的是减小参数，防止过分拟合数据集，一般认为，模型参数越小，模型越简单，反之越复杂。目前权重衰减的技术的实现都是在优化器上面，torch.optim.SGD(args) 中含有 weight_decay 作为参数输入。 
    + 参考链接： https://blog.csdn.net/zhaohongfei_358/article/details/129625803
+ Perplexity
    + PPL 是用在自然语言处理领域（NLP）中，衡量语言模型好坏的指标。它主要是根据每个词来估计一句话出现的概率，并用句子长度做 normalize
    + PPL 越小， p(wi) 越大，一句我们期望的 sentence 出现的概率就越高。
    + 参考链接：https://blog.csdn.net/index20001/article/details/78884646

+ 优化器的作用
    + 用来更新和计算影响模型训练和模型输出的网络参数,使其逼近或达到最优值，从而最小化损失函数。
    + 参考链接：https://aistudio.baidu.com/aistudio/projectdetail/4411881

+ 梯度爆炸
    + 参考链接：https://baijiahao.baidu.com/s?id=1587462776558160221&wfr=spider&for=pc

#### 3.1.2 Demo 源代码学习
+ 1. 初始化 device
```
 if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
```

+ 2. 初始化 tokenizer
```
# 根据模型名字调用 transformer 库中的 AutoTokenizer.from_pretrained 方法。
if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, fast_tokenizer=fast_tokenizer)
```
tokenizer 的使用可以参考：
https://blog.csdn.net/starzhou/article/details/117409414

+ 3. 加载预训练模型
```
model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)
```

+ 4. 数据准备
    + 切分数据集，切分数据集对应不同阶段的训练
    + 对数据集进行 shuffle，打乱顺序
    + 获得打乱顺序的数据集子集，对该数据集进行处理(使用标记器对每个 sentence进行处理)
    ```
    def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                chosen_dataset.append(chosen_token)
    ```
    + 根据训练的不同阶段，返回的数据集输入不太一样
    ```
    class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id

    ```

+ 5. 创建 dataloader
    + 初始化采样器
    ```
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    ```
    + 创建 dataloader
    ```
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    ```
+ 6. 选择优化器优化策略
    ```
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    ```

+ 7. 设置动态调整学习率策略
```
   lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )  
```
    + 参考链接：https://zhuanlan.zhihu.com/p/466992867

+ 8. deepspeed 初始化
```
  model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
```
+ 9. 模型训练
```
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        # 在使用 pytorch 构建神经网络的时候，训练过程中会在程序上方添加一句 model.train()，作用是：启用 
        # batchnorm 和 dropout。如果模型中有 BN 层和 DropOut，需要在训练时添加 model.train().
        # model.train() 是保证 BN 层能够用到每一批数据的均值和方差。 对于 Dropout, model.train() 是随机选# 取一部分网络连接来训练更新参数

        model.train() 
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            model.backward(loss)
            model.step()

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        model.tput_timer.update_epoch_count()


    def evaluation(model, eval_dataloader):
        # model.eval() 的作用是 不启用 batch_norm 和 dropout
        # 如果模型中有 BN 层 和 dropout，在测试时需要添加 model.eval()
        # model.eval() 是保证 BN 层能够用全部训练数据的均值和方差。
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity
```

### 3.2 奖励模型微调
#### 3.2.1 概念
+ 奖励模型微调 
    + 使用一个包含人类对同一查询（query）的多个答案打分的数据集来训练一个独立的（通常比 SFT 小的）奖励模型 (RW)

### 3.3 RLHF 训练
#### 3.3.1 概念
+ 奖励模型微调 
    + 利用 Proximal Policy Optimization（PPO）算法, 根据 RW 模型的奖励反馈进一步微调 SFT 模型。