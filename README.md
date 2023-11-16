# VGG16 with PyTorch

*Authors: hyf493*

Note: 所有脚本需要在 `vgg16`.文件夹下运行

## 依赖

建议使用python3和虚拟环境

```
pip install -r requirements.txt
```

## 任务

给出一只手做着代表 0、1、2、3、4 或 5 的手势的图像，预测正确的标签。（分类）


## 下载 SIGNS 数据集

在vgg16示例中，我们将使用 SIGNS 数据集。该数据集托管在百度网盘上，请下载链接：https://pan.baidu.com/s/1x4Pw_mIpd8GIvLBjUcidzQ    提取码：yq84

这将下载转换为64*64大小的SIGNS 数据集，其中包含手势组成 0 到 5 之间数字的照片。
以下是数据结构：

```
64x64_SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

图片的命名方式为"{label}_IMG_{id}.jpg"，其中标签为"[0, 5]"。
训练集包含 1,080 幅图像，测试集包含 120 幅图像。

将数据集移至 `data` 中。

## Quickstart

1.__experiment__     `experiments` 目录下创建了一个 `base_model` 目录。其中包含一个文件 `params.json` ，用于设置实验的超参数。

```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
2.__Train__ 

```
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
它将实例化一个模型，并按照 `params.json` 中指定的超参数在训练集上对其进行训练。它还会在验证集上评估一些指标。

3.在测试集上进行评估，根据验证集上的性能运行多次实验并选出最佳模型和超参数后，就可以最终评估模型在测试集上的性能了。运行

```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```

## CSDN

我的csdn主页：https://blog.csdn.net/qq_53909832?spm=1000.2115.3001.5343

