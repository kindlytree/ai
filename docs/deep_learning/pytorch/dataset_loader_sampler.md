# Dataset, DataLoader, Sampler

# datasets

## References
- Dataset, DataLoader, Sampler三者之间的关系
    - https://www.cnblogs.com/marsggbo/p/11308889.html


## build_dataset

```
data_loaders = [
    build_dataloader(
        ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
    for ds in dataset
]

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
    pin_memory=False,
    **kwargs)    

    
    
```

## 自定义Dataset
我们看一下代码，自定义Dataset只需要最下面一个class,继承自Dataset类。有三个私有函数

def __init__(self, loader=default_loader):

这个里面一般要初始化一个loader(代码见上面),一个images_path的列表，一个target的列表

def __getitem__(self, index)：

这里吗就是在给你一个index的时候，你返回一个图片的tensor和target的tensor,使用了loader方法，经过 归一化，剪裁，类型转化，从图像变成tensor

def __len__(self):

return你所有数据的个数

这三个综合起来看呢，其实就是你告诉它你所有数据的长度，它每次给你返回一个shuffle过的index,以这个方式遍历数据集，通过 __getitem__(self, index)返回一组你要的（input,target）

## Dataloader
实例化一个dataset,然后用Dataloader 包起来
```
train_data  = trainset()
trainloader = DataLoader(train_data, batch_size=4,shuffle=True)
```

## Sampler
