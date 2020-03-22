# autograd

## References
- https://www.cnblogs.com/Thinker-pcw/p/9630367.html

## 简介

Pytorch中神经网络包中最核心的是autograd包，我们先来简单地学习它，然后训练我们第一个神经网络。

autograd包为所有在tensor上的运算提供了自动求导的支持，这是一个逐步运行的框架，也就意味着后向传播过程是按照你的代码定义的，并且单个循环可以不同

我们通过一些简单例子来了解

Tensor
torch.tensor是这个包的基础类，如果你设置.requires_grads为True，它就会开始跟踪上面的所有运算。如果你做完了运算使用.backward()，所有的梯度就会自动运算，tesor的梯度将会累加到.grad这个属性。

若要停止tensor的历史纪录，可以使用.detch()将它从历史计算中分离出来，防止未来的计算被跟踪。

 为了防止追踪历史（并且使用内存），你也可以将代码块包含在with torch.no_grad():中。这对于评估模型时是很有用的，因为模型也许拥有可训练的参数使用了requires_grad=True,但是这种情况下我们不需要梯度。

还有一个类对autograd的实现非常重要，——Function

Tensor和Function是相互关联的并一起组成非循环图，它编码了所有计算的历史，每个tensor拥有一个属性.grad_fn,该属性引用已创建tensor的Function。（除了用户自己创建的tensor,它们的.grad_fn为None）。

如果你想计算导数，可以在一个Tensor上调用.backward()。如果Tensor是一个标量（也就是只包含一个元素数据），你不需要为backward指明任何参数，但是拥有多个元素的情况下，你需要指定一个匹配维度的gradient参数。

## 实例：


```
grad_lane_wrt_param = torch.autograd.grad(loss, \
                            self.shared_parameter, retain_graph=True, create_graph=True)
glwp = torch.norm(grad_lane_wrt_param[0], 2)

```