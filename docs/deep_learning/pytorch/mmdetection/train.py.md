# train.py

## build_detector
- file path is  ./mmdet/models/builder.py
- build_detector function in builder.py
- from mmdet.utils import build_from_cfg
- build_from_cfg函数的真正调用的地方在./mmdet/utils/registry.py,主要是通过cfg生成对应的对象

### train_detector
- mmdet/apis/train.py



### 使用方法

#### Train

```
python tools/train.py configs/multi_task/multi_task_prototype.py --no-validate
python tools/test.py config_file checkpoint_file
```