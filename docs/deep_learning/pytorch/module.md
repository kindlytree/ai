# nn.Module

## nn.ModuleList, nn.Sequential
```
class Conv2dBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv2dBatchRelu, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii/2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size/2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

det_subnet = [
    OrderedDict([
        ('feature_map_32d_conv', Conv2dBatchRelu(512, 512, 3, 1)),
        ('feature_map_32d', nn.Conv2d(512, self.num_anchors * (5 + self.cls_channels), 1, 1, 0, bias=False)),
    ]),

    OrderedDict([
        ('feature_map_16d_conv', Conv2dBatchRelu(256, 512, 3, 1)),
        ('feature_map_16d', nn.Conv2d(512, self.num_anchors * (5 + self.cls_channels), 1, 1, 0, bias=False)),
    ]),

    OrderedDict([
        ('feature_map_8d_conv', Conv2dBatchRelu(256, 512, 3, 1)),
        ('feature_map_8d', nn.Conv2d(512, self.num_anchors * (5 + self.cls_channels), 1, 1, 0, bias=False)),
    ]),
]

self.det_subnet = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in det_subnet])

def forward(self, x):
    features = list(x)

    feature_map_32d = self.det_subnet[0](features[-1])
```