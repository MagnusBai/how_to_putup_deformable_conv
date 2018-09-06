# 关于坐标偏移参数的工程化实现讨论

## 1. 背景

参加face++面试时，遇到问题：
```
mask-rcnn中roialign pooling层，与 deformable conv层有什么相同、异同？
```
现在的答案如下：

  * 相同点：对输入层feature map中，在 [h, w] 索引范围内，用线性插值的方法，得到某连续的值坐标 [h.f, w.f] 所对应的feature。
  
  * 不同点：deformable conv的实现中，坐标值偏移量是可学习的。而roialign中，连续坐标是在网络设置好后固定的。

于是自己产生问题：`deformable conv中的带偏移的参数，如何训练、工程化？`

## 2. 可学习的坐标偏移参数如何实现工程化

以Deformable Convolution Layer为例（DConv层）。

![im](https://i.imgur.com/AjDxteU.png)

参考1. 2014年Ross Girshick的 "Deformable Part Model are CNN" 中的distance transform pooling
参考2. 2017年Jifeng Dai的 "Deformable Convolutional Networks"（后面简称DCN）中的deformable convolution

### 2.1 基本思路

DCN中 Figure 2的过程，是DConv-layer的输出效果：输出的feature map中的某个feature，打破了输入层kernel的空间限制，具有位置偏移的能力。

基本思路是用3个layer组成DConv-layer：
  * layer1： 采用一般卷积 ;
  * layer2： 采用depthwise卷积获得x,y连续值偏移量 （包含1.卷积层的latent parameter，2. 量化为特定连续x,y的real parameter，3.一个latent<->real 的 **双向量化转换函数** ）（卷积的kernel_size = 2*最大偏移量）;
  * layer3： 输入为：1.第二层的x, y连续值偏移量 2.第一层的featuremap，输出为：在该坐标值下


### 2.2 训练方式

layer2, layer3，可以合并为一套Conv层反向传递模型，

第[i+1]轮的损失函数BP信号`∂(loss)/∂(latent_parameter[i])`，传递到layer2+3，为latent_parameter[i+1]

量化latent parameter[i+1]为real parameter[i+1]，

根据`∂(loss)/∂(real_parameter[i+1])`，将损失函数BP信号，传递至layer1

### 2.3 部署方式

#### 初步优化的思路：

layer2,3的部署：
根据最终layer 2，3的训练参数，推算出layer3输出，与layer1输出的对应关系（每个输出值，对应4个layer1输入坐标，以及4个权重，通过查表进行导出）；

#### 进一步优化的思路：

根据layer3, layer1元素的关系，改写layer1 Conv实现中im2col()的过程，提升初步优化中，查表操作的可并行性。

-------------

ps.面试题中svm中最终优化损失函数Hinge Loss，与NN中的softmax loss(cross-entropy loss)差异，直观比较的文章：
[神经网络的分类模型 LOSS 函数为什么要用 CROSS ENTROPY](http://jackon.me/posts/why-use-cross-entropy-error-for-loss-function/)



