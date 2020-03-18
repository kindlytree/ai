# Multi-Task Learning

## References
- Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics(CVPR2017)
  - 论文：https://arxiv.org/abs/1705.07115v3
  - 代码：https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb 
  - 应用:视觉计算，场景理解
  - 简要:用统计概率的思想解释权重，并学习权重
- Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks(ICML2018)
  - 论文: https://arxiv.org/pdf/1711.02257.pdf
  - code: https://github.com/hosseinshn/GradNorm/blob/master/GradNormv10.ipynb
  - https://zhuanlan.zhihu.com/p/100555359
  - 简要：用不同loss相对于共享参数的梯度以及学习的速率等相关指标进行权重的学习，消耗内存和计算资源较大
- Multi-Task Learning as Multi-Objective Optimization
  - 采用梯度方向的概念和相关理论证明来进行权重的调整 
- End-to-End Multi-Task Learning with Attention(CVPR2019)
  - https://github.com/lorenmt/mtan
  - https://blog.csdn.net/qq_21157073/article/details/98884819
  - https://arxiv.org/pdf/1803.10704v1.pdf