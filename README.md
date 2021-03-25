# 从 YoloV3 中进阶 Pytorch

pytorch == 1.6.0（nms 需要）

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
16. as_tensor vs from_numpy
17. numpy 中 array 和 asarray 的区别：array和asarray都可以将结构数据转化为ndarray，但是主要区别就是当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，但asarray不会。
18. torch.max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)

> Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. And indices is the index location of each maximum value found (argmax).
> If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1. Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensors having 1 fewer dimension than input.

19. cpu().unique() 待记
20. multiprocessing

```python
    # TODO Note: 必须在主函数里，这样，之后才能 fork 多线程
    # 为使用了 multiprocessing  的程序，提供冻结以产生 Windows 可执行文件的支持。
    # 需要在 main 模块的 if __name__ == '__main__' 该行之后马上调用该函数。
    # 由于Python的内存操作并不是线程安全的，对于多线程的操作加了一把锁。这把锁被称为GIL（Global Interpreter Lock）。
    # 而 Python 使用多进程来替代多线程
    # torch.multiprocessing.freeze_support()
    #
    # torch.manual_seed(1)  # fake random makes reproducible
```

21. DataLoader在数据集上提供单进程或多进程的迭代器，几个关键的参数意思：
    - shuffle：设置为True的时候，每个世代都会打乱数据集
    - collate_fn：如何取样本的，我们可以定义自己的函数来准确地实现想要的功能
    - drop_last：告诉如何处理数据集长度除于batch_size余下的数据。True就抛弃，否则保留
    - num_workers=0,  # 表示开启多少个线程数去加载你的数据，默认为0，代表只使用主进程

22. BCHW 还是 BCWH，RGB 还是 BGR
    - OpenCV默认通道为BGR，PIL 是 RGB
    - Pytorch 使用 RGB
    - BCHW GPU 训练
    - ToTensor：ToTensor()接收PIL格式的数据, 或者是直接从PIL转来的np.ndarray格式数据, 只要保证进来的数据取值范围是[0, 255], 形状是[h, w, c], 像素顺序是RGB, 它就会帮你做下面的事情
        - 取值范围[0, 255] / 255.0 => [0, 1.0], 数据格式从int8变成了float32
        - 形状(shape)转为[c, h, w]
        - 像素顺序依旧是RGB
        - [Pytorch数据前后处理整理](https://www.jianshu.com/p/c0ba27e392ff)

作者：conson_wm
链接：https://www.jianshu.com/p/c0ba27e392ff
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

> 问题：为什么深度学习中普遍用BRG描述图像，而非RGB通道？
> 答1：因为caffe，作为最早最流行的一批库的代表，用了opencv，而opencv默认通道是bgr的。这是opencv的入门大坑之一，bgr是个历史遗留问题，为了兼容早年的某些硬件。其实你自己训练完全可以用rgb，新库也基本没了bgr还是rgb这个问题，就是切换下顺序。但如果你要用一些老的训练好的模型，就得兼容老模型的bgr。
> 答2：因为OpenCV默认通道为BGR，至于为什么用BGR，可能是因为当时比较流行BGR，以至于后来RGB变为主流后，也不方便改了，就一直沿用BGR。而caffe又是用了opencv的，所以没办法。智能外部转换一下。

23. PIL image size

```python
from PIL import Image 
im = Image.open('whatever.png') 
width, height = im.size
```

24. np.asarray(PIL.Image.Image)， WRITEABLE : False
    - ndarray.flags

25. only one element tensors can be converted to Python scalars
    - list(tensor) 无法 torch.as_tensor
    - ndarray(tensor) 无法 torch.from_numpy

26. The given NumPy array is not writeable
    - It is caused by img = transforms.ToTensor()(img)
