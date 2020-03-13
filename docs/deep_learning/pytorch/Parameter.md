# Parameter

## nn.Paramter是可以更新的类，是Variable的子类
```
self.register_parameter('loss_weights',nn.Parameter(Variable(torch.zeros(5).cuda(), requires_grad=True)))
loss_det[idx] = torch.exp(-self.loss_weights[idx])* loss_det[idx] + self.loss_weights[idx]
```