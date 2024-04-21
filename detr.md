# object detection
论文来自：https://zhuanlan.zhihu.com/p/388975066

## up-detr（unsupervised-pretrained-detection-transformer）:
+ https://medium.com/analytics-vidhya/up-detr-unsupervised-pre-training-for-object-detection-with-transformers-a-review-c4b996e12a9c
+ 无监督预训练：图像随机裁剪patches作为query输入transformer预训练
+ 输入图像通过cnn得到特征图f，f输入位置编码，再输出到transformer编码器解码器；
+ 同一张输入图像的随机裁剪patches经过全局平均池化输入cnn得到补丁的特征图p，再和上面一样大小的query输入到解码器

## detr: 
+ https://medium.com/@faheemrustamy/detection-transformer-detr-vs-yolo-for-object-detection-baeb3c50bc3
+ backbone：删去fc层（全连接）的resnet50，因为要换头，先提取图像的特征
+ 然后再转换成transformer的输入格式，输入transformer（理解图像不同区域之间的关系（attention））使用自注意力机制（捕捉输入数据不同部分的关系，并给不同部分打分）
+ 位置编码（cnn具有空间感知能力但transformer没有）告诉transformer特征对应的位置
+ 预测head：类别预测和框预测

## ore（open world object detection）：
+ 检测出unknown目标；且不忘记之前学到的目标
+ 基于faster-rcnn
