# 源码学习记录

## diffusion model
symbolic regression: 用一个数学公式来描述数据

2023plrdiff: 先从LRHS图像中提取关键信息到系数矩阵E，将LRHS、PAN（清晰度更高）、E作为输入来学习；使用diffusion model ，构建出一个基础张量A（我们的重建图像），然后将A和E结合，得到最终结果

diffusion model：将一种图片逐步加高斯噪声，渐渐变成模糊的图片，然后输入模型学习，使得模型学会将图片复原

2018sts-cnn：adam+bp reconstruction

原数据集和sr数据集之间的相似程度：
compute fid score: 
https://github.com/mseitzer/pytorch-fid

sr方法：SR3架构类似于unet
https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement

unet（左边编码压缩逐渐模糊，右边解码扩展逐渐清晰，跳跃连接左右将左边的关键信息传递给右边，在语义分割中，unet的最后一层进行像素级分类）

sr方法：codi
https://github.com/fast-codi/CoDi

PGCU：pansharpening
lrms图像（低分辨率多光谱）pan图像（高分辨率单光谱）论文看不懂（大概是先分别对两种图片提取信息，提取后融合在一起，分为三个通道输出到第二步进行学习分布，最后一步就是做微调）

## object detection
论文来自：https://zhuanlan.zhihu.com/p/388975066

up-detr（unsupervised-pretrained-detection-transformer）:
+ https://medium.com/analytics-vidhya/up-detr-unsupervised-pre-training-for-object-detection-with-transformers-a-review-c4b996e12a9c
+ 无监督预训练：图像随机裁剪patches作为query输入transformer预训练
+ 输入图像通过cnn得到特征图f，f输入位置编码，再输出到transformer编码器解码器；
+ 同一张输入图像的随机裁剪patches经过全局平均池化输入cnn得到补丁的特征图p，再和上面一样大小的query输入到解码器

detr: 
+ https://medium.com/@faheemrustamy/detection-transformer-detr-vs-yolo-for-object-detection-baeb3c50bc3
+ backbone：删去fc层（全连接）的resnet50，因为要换头，先提取图像的特征
+ 然后再转换成transformer的输入格式，输入transformer（理解图像不同区域之间的关系（attention））使用自注意力机制（捕捉输入数据不同部分的关系，并给不同部分打分）
+ 位置编码（cnn具有空间感知能力但transformer没有）告诉transformer特征对应的位置
+ 预测head：类别预测和框预测

ore（open world object detection）：
+ 检测出unknown目标；且不忘记之前学到的目标
+ 基于faster-rcnn








