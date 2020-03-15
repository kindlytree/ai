# Basics

## Tensor

### basics
print(pytorch_tensor1.shape)
print(pytorch_tensor1.size())
print(pytorch_tensor1.type()) #得到tensor的数据类型
print(pytorch_tensor1.dim()) #得到tensor的维度
print(pytorch_tensor1.numel()) #得到tensor所有元素的个数

### permute
permute可以对任意高维矩阵进行转置.
但没有 torch.permute() 这个调用方式， 只能 Tensor.permute()。
t.rand(2,3,4,5).permute(3,2,0,1).shapeOut[669]: torch.Size([5, 4, 2, 3])

### transpose
transpose只能操作2D矩阵的转置。有两种调用方式。
连续使用transpose也可实现permute的效果。
t.rand(2,3,4,5).transpose(3,0).transpose(2,1).transpose(3,2).shape
Out[672]: torch.Size([5, 4, 2, 3])

### contiguous
contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。 
一种可能的解释是： 
有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。 

## 函数

### F.softmax
b = F.softmax(input,dim=0) # 按列SoftMax,列和为1
F.softmax(input,dim=1)   # 按行SoftMax,行和为1

### squeeze()和unsqueeze()
torch.unsqueeze(gt_semantic, 1)