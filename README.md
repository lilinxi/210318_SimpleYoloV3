# 从 YoloV3 中进阶 Pytorch

1. bias=False，卷积之后，如果要接 BN 操作，最好是不设置偏置，因为不起作用，而且占显卡内存
2. 使用 LeakyReLU，收敛速度快，梯度不饱和不消失，负数区域神经元也不会死掉
3. BCHW 和 BHWC：设计网络时充分考虑两种格式，最好能灵活切换，在 GPU 上训练时使用 NCHW 格式，在 CPU 上做预测时使用 NHWC 格式。
4. 交叉熵损失函数
5. BN：这三步就是我们在刚刚一直说的 normalization 工序, 但是公式的后面还有一个反向操作, 将 normalize 后的数据再扩展和平移. 原来这是为了让神经网络自己去学着使用和修改这个扩展参数 gamma, 和 平移参数 β, 这样神经网络就能自己慢慢琢磨出前面的 normalization 操作到底有没有起到优化的作用, 如果没有起到作用, 我就使用 gamma 和 belt 来抵消一些 normalization 的操作.
    - default momentum = 0.1，代表当前 batch 产生的统计数据的重要性为 0.1，历史数据为 0.9
6. 修改模型权重的名称：

```python
from collections import OrderedDict
new_dict = OrderedDict()
for key in model.state_dict():
	if key == "features.0.weight":
		new_dict["features.0.inner_conv2d.weight"] = model.state_dict()[key]
		new_dict["features.0.weight"] = model.state_dict()[key]
	elif key == "features.0.bias":
		new_dict["features.0.inner_conv2d.bias"] = model.state_dict()[key]
		new_dict["features.0.bias"] = model.state_dict()[key]
	elif key == "features.4.weight":
		new_dict["features.4.inner_conv2d.weight"] = model.state_dict()[key]
		new_dict["features.4.weight"] = model.state_dict()[key]
	elif key == "features.4.bias":
		new_dict["features.4.inner_conv2d.bias"] = model.state_dict()[key]
		new_dict["features.4.bias"] = model.state_dict()[key]
	elif key == "features.8.weight":
		new_dict["features.8.inner_conv2d.weight"] = model.state_dict()[key]
		new_dict["features.8.weight"] = model.state_dict()[key]
	elif key == "features.8.bias":
		new_dict["features.8.inner_conv2d.bias"] = model.state_dict()[key]
		new_dict["features.8.bias"] = model.state_dict()[key]
	elif key == "features.12.weight":
		new_dict["features.12.inner_conv2d.weight"] = model.state_dict()[key]
		new_dict["features.12.weight"] = model.state_dict()[key]
	elif key == "features.12.bias":
		new_dict["features.12.inner_conv2d.bias"] = model.state_dict()[key]
		new_dict["features.12.bias"] = model.state_dict()[key]
	else:
		new_dict[key] = model.state_dict()[key]
```
7. 权值初始化：https://blog.csdn.net/hyk_1996/article/details/82118797
8. nn.modulelist 和 nn.sequential：https://zhuanlan.zhihu.com/p/64990232
    - nn.modulelist 不同于 list
    - nn.modulelist 无 forward
9. 卷积网络中的感受野：https://www.cnblogs.com/pprp/p/12346759.html
10. num_batches_tracked: 如果没有指定momentum, 则使用1/num_batches_tracked 作为因数来计算均值和方差(running mean and variance).
11. nn.DataParallel 的坑
12. tensor.detach() 和 tensor.data: https://zhuanlan.zhihu.com/p/67184419
13. Tensor 和 Variable：https://zhuanlan.zhihu.com/p/34298983
14. Pytorch 中的 Tensor , Variable和Parameter区别与联系：https://blog.csdn.net/u014244487/article/details/104372441
15. contiguous，如果想要变得连续使用contiguous方法，如果Tensor不是连续的，则会重新开辟一块内存空间保证数据是在内存中是连续的，如果Tensor是连续的，则contiguous无操作。
