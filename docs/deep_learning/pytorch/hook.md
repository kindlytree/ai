# pytorch hooks

## References
- [保留中间变量的导数](https://blog.csdn.net/qq_42110481/article/details/81043932)
- [pytorch怎么计算中间层的特征或梯度](https://www.cnblogs.com/Wanggcong/p/10269823.html)

## Samples

```
def grad_hook(self, module, grad_input, grad_output):
    #print(module, module.name)
    for name, module_ in self.named_modules():
        if name in self.grad_buffer_map.keys() and module_ == module:
            ##print(grad_output)
            #print(self.grad_buffer_map)
            #print(torch.mean(torch.abs(grad_output)))
            #print(len(grad_output))
            #print(name, torch.mean(torch.abs(grad_output[0])))
            ##print('*********************', grad_output[0].size())
            if len(self.grad_buffer_map[name]) >= 10:
                self.grad_buffer_map[name][self.iter_ %10] = torch.mean(torch.abs(grad_output[0])).item()
            else:
                self.grad_buffer_map[name].append(torch.mean(torch.abs(grad_output[0])).item())


def init_hooks(self):
    for name, module in self.named_modules():
        if name in self.grad_buffer_map.keys():
            module.register_backward_hook(self.grad_hook)

```
