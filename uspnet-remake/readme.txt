Net文件夹中：
自定义的模块包括LSTM（修改attention_net_with_w函数）

信号肽（SP）是位于蛋白质N末端的短肽。
对信号肽进行分类并预测切割位点位置。解决了数据不平衡问题（xxx）
同时不依赖additional group information of proteins（因为这些信息不一定有用）
USPNet 仅用原始氨基酸序列和大型蛋白质语言模型就可以学习 SP 结构，从而能够有效地发现远离现有知识的新型 SP
解决数据不平衡问题：类平衡损失与标签分布感知边缘（LDAM）损失结合起来作为USPNet的损失函数
是端到端的，仅以原始氨基酸作为输入。

basemodel：Bi-LSTM
流程：
1.氨基酸序列->Lx20矩阵（L为序列长度）
2.将1的序列进入特征提取模块+embedding layer
3.将2的结果输入bi-lstm（self attention的bi-lstm+cnn）同时提取前向后向的依赖关系、全局特征、局部特征
4.将3的结果输入一个head（基于mlp）来预测：切割位点、sp类别
