# Parameter

## nn.Paramter是可以更新数值的类，是Variable的子类
```
self为nn.Module的子类
self.register_parameter('loss_weights',nn.Parameter(Variable(torch.zeros(5).cuda(), requires_grad=True)))
loss[idx] = torch.exp(-self.loss_weights[idx])* loss[i] + self.loss_weights[idx]
```