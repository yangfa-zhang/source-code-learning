#  Language model, Sequence and function prediction
  
## USPNet  
#### task 
  氨基酸序列作为输入，预测切割位点和SP（signal peptide信号肽）的类别
#### motivation
1. data imbalance problem
2. 以往的方法利用additional group information of proteins来提高性能，然而这些proteins不总是available
#### technical contribution
1. lstm+attention作为训练模型
2. 类平衡损失与标签分布感知边缘（LDAM）损失结合起来作为USPNet的损失函数，解决data imbalance problem
#### pipeline
basemodel：Bi-LSTM  
流程：  
1. 氨基酸序列->Lx20矩阵（L为序列长度）（生成msa，然后再输入msa transformer）
2. 将1的序列进入特征提取模块+embedding layer
3. 将2的结果输入bi-lstm（self attention的bi-lstm+cnn）同时提取前向后向的依赖关系、全局特征、局部特征
4. 将3的结果输入一个head（基于mlp）来预测：切割位点、sp类别


## Msa Transformer
#### motivation  
#### technical contribution
1. 行、列注意力操作
- 行注意力：学习msa不同序列的相似之处
- 列注意力：全局信息，学习结构信息
- O(M^2L^2)到O(ML^2) + O(LM^2)：  
   M是序列数量，L是序列长度，O(M^2L^2）每个序列内部计算attention_w需要L^2, 序列之间的计算需要M^2；O(ML^2) + O(LM^2) ，行注意力机制中，M列只计算内部的attention_w，需要M*L^2，列注意力机制中，以相似的操作，需要L*M^2
2. 使用mask训练
- 使用无标注的数据进行训练
3. 跨越多个不同的蛋白质家族进行训练
- 使训练结果在样本外同样表现良好
#### pipeline


## ProtENN
#### motivation  
#### technical contribution

## RNA-FM
#### motivation  
#### technical contribution
