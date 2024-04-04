## 代码内容
Net文件夹中：
自定义的模块包括LSTM（修改attention_net_with_w函数）

## 论文内容
- 信号肽（SP）是位于蛋白质N末端的短肽。
- 对信号肽进行分类并预测切割位点位置。解决了数据不平衡问题（xxx）
- 同时不依赖additional group information of proteins（因为这些信息不一定有用）
- USPNet 仅用原始氨基酸序列和大型蛋白质语言模型就可以学习 SP 结构，从而能够有效地发现远离现有知识的新型 SP
- 解决数据不平衡问题：类平衡损失与标签分布感知边缘（LDAM）损失结合起来作为USPNet的损失函数
- 是端到端的，仅以原始氨基酸作为输入。

## 模型流程
basemodel：Bi-LSTM  
流程：  
1. 氨基酸序列->Lx20矩阵（L为序列长度）（生成msa，然后再输入msa transformer）
2. 将1的序列进入特征提取模块+embedding layer
3. 将2的结果输入bi-lstm（self attention的bi-lstm+cnn）同时提取前向后向的依赖关系、全局特征、局部特征
4. 将3的结果输入一个head（基于mlp）来预测：切割位点、sp类别

## 对lstm加attention细节
1. lstm输出分割为两部分后相加得到h
2. 对h计算注意力权重atten_w
3. 将h映射到m
4. 对atten_w转换为softmax_w
5. h与softmax_w相乘得到context，提取了上下文依赖信息
6. context加权平均得到result
7. result加一个dropout层

## data imbalance：
- 训练数据中不同类别的样本数量差异
- 非信号肽序列（不含sp的氨基酸序列），包含少数类别的sp的序列数量远小于非信号肽序列

### 解决data imbalance的方法包括re sampling和re weighting
- resampling：减少数量较多的类别or复制或生成的方式增加数量较少的类别的数据量
- reweighting：为每个类别分配权重

### ldam loss介绍：
- 参考：https://zhuanlan.zhihu.com/p/308298563
- margin（第i类样本到决策边界的最小值）
- 最佳的margin rj=C除以 nj的1/4次方  C是未定超参数（这个公式有数学推导）

### uspnet的创新点：
- logits（模型输出的未归一化的预测值；logits是模型对每个类别的预测强度，如果模型对预测准确有信心，logits就越高）
- 分类器的最后一层加入归一化线性层，归一化后得到每一个类别的权重向量，agent vector就是这些权重向量，
- 代理向量捕捉了类别的本质特征（因此具有更好的泛化能力）

## msa transformer的添加
- 对每一个氨基酸序列使用searching updated UniClust30 with HHblits方法生成msa数据
- msa数据直接输入msa transformer预训练模型
- msa transformer的生成embedding用于输入uspnet
- msa transformer能够学习到序列之间的相似性，相似性意味着有相似的结构和功能
