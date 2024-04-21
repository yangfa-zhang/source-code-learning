# diffusion model
diffusion model：将一种图片逐步加高斯噪声，渐渐变成模糊的图片，然后输入模型学习，使得模型学会将图片复原

## symbolic regression: 用一个数学公式来描述数据

## reconstruction
+ 2023plrdiff: 先从LRHS图像中提取关键信息到系数矩阵E，将LRHS、PAN（清晰度更高）、E作为输入来学习；使用diffusion model ，构建出一个基础张量A（我们的重建图像），然后将A和E结合，得到最终结果
+ 2018sts-cnn：adam+bp reconstruction
+ 原数据集和sr数据集之间的相似程度：compute fid score: https://github.com/mseitzer/pytorch-fid

## super resolution方法：
+ SR3架构类似于unet: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement  
  unet（左边编码压缩逐渐模糊，右边解码扩展逐渐清晰，跳跃连接左右将左边的关键信息传递给右边，在语义分割中，unet的最后一层进行像素级分类）
+ codi: https://github.com/fast-codi/CoDi
+ PGCU：pansharpening  
lrms图像（低分辨率多光谱）pan图像（高分辨率单光谱）论文看不懂（大概是先分别对两种图片提取信息，提取后融合在一起，分为三个通道输出到第二步进行学习分布，最后一步就是做微调）
