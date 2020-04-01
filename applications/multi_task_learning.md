# Multi-Task Learning

## References

### Paper&code
- Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics(CVPR2017)
  - 论文：https://arxiv.org/abs/1705.07115v3
  - 代码：https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb 
  - code:https://github.com/ranandalon/mtl
<<<<<<< HEAD
  - code:https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
=======
  - code:https://github.com/oscarkey/multitask-learning
>>>>>>> 83214380660d5b3fd6c827320cb8f0b3a5d74f7f
  - 应用:视觉计算，场景理解
  - 简要总结:用统计概率的思想解释权重，即训练样本服从联合概率最大化，并学习权重
- Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks(ICML2018)
  - 论文: https://arxiv.org/pdf/1711.02257.pdf
  - code: https://github.com/hosseinshn/GradNorm/blob/master/GradNormv10.ipynb
  - code: https://github.com/hav4ik/Hydra/blob/master/src/applications/trainers/gradnorm.py
  - https://zhuanlan.zhihu.com/p/100555359
  - 简要总结：用不同loss相对于共享参数的梯度以及学习的速率等相关指标进行权重的学习，消耗内存和计算资源较大
- Multi-Task Learning as Multi-Objective Optimization
  - 简要总结：采用梯度方向的概念和相关理论证明来进行权重的调整
  - code: https://github.com/intel-isl/MultiObjectiveOptimization
  - code: https://github.com/hav4ik/Hydra
- End-to-End Multi-Task Learning with Attention(CVPR2019)
  - https://github.com/lorenmt/mtan
  - https://blog.csdn.net/qq_21157073/article/details/98884819
  - https://arxiv.org/pdf/1803.10704v1.pdf
  - 简要总结：用Attention的思想，训练和推理应该消耗GPU容量会比较大
- Dynamic Task Prioritization for Multitask Learning
  - http://svl.stanford.edu/assets/papers/guo2018focus.pdf
  - 简要总结：通过任务的难易程度，给予困难的任务以较高优先级
- Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory
  - https://github.com/jkokkin/UberNet
  - https://arxiv.org/pdf/1609.02132.pdf 
- Training Complex Models with Multi-Task Weak Supervision
  - code: https://github.com/HazyResearch/metal 
  - https://arxiv.org/pdf/1810.02840.pdf
- Stochastic Filter Groups for Multi-Task CNNs: Learning Specialist and Generalist Convolution Kernels
  -  https://arxiv.org/abs/1908.09597
- Learning Sparse Sharing: Architectures for Multiple Tasks **AAAI 2020**
  - https://arxiv.org/abs/1911.05034
  - https://github.com/choosewhatulike/sparse-sharing
  - 简要总结：每一个任务训练一个子网络（学到稀疏矩阵），最后多个任务联合训练，不同的任务用不同的子网络训练，优点：可以不用所有的样本都要标记所有的任务.
- ICCV2019 oral：Many Task Learning With Task Routing
  - https://github.com/gstrezoski/TaskRouting
  - https://arxiv.org/abs/1903.12117
  - 简要总结：每个任务都有专门的chanel掩码来确定子网络，不同的任务可以部分共享channel，不利的地方就是inference的时候要单个任务单独inference，时间上没有多少优势  
## Projects
- https://github.com/hav4ik/Hydra  Multi-Task Learning Framework on PyTorch