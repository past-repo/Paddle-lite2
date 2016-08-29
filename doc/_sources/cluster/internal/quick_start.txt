# 快速入门

下面通过MNIST和AlexNet多机训练的例子来展示新平台的最基本使用方法。

## MNIST

- 准备配置文件`$PADDLE_PLATFORM/test/mnist/mnist_cluster_job.py`

  - 配置集群相关的参数
  ```python
  cluster_config(
      fs_name = "hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
      fs_ugi = "paddle_demo,paddle_demo",
      work_dir ="/app/idl/idl-dl/paddle/cluster_test/mnist/",
      )
  ```

  `cluster_config`配置了HADOOP集群的文件系统名称`fs_name`，`ugi`，以及数据集模型存放的父目录`work_dir`。训练数据的路径为`$work_dir/train_data`，测试数据的路径为`$work_dir/test_data`，模型输出的路径为`$work_dir/output_时间戳`。更详细的介绍见[集群配置文件](../cluster_config.html#config-file)。

  - 配置训练数据、优化算法、以及网络结构：
  ```python
  """ 数据配置 """
  TrainData(ProtoData(files = "train.list"))

  """ 算法配置 """
  settings(learning_rate=1e-3,
           learning_method=AdamOptimizer(),
           batch_size=1000)
           
  """ 网络结构 """
  img = data_layer(name='input', size=784)
  hidden = fc_layer(input=img, size=800)
  hidden = fc_layer(input=hidden, size=800)
  prediction = fc_layer(input=hidden, size=10, act=SoftmaxActivation())
  outputs(classification_cost(input=prediction,
                              label=data_layer(name='label', size=10)))
  ```
  平台在训练过程中会定时使用测试集对最新模型进行评估, 通常情况下建议开启测试以实时观察模型在测试集上的表现。只需将测试数据放置在`$work_dir/test`下，并在`TrainData(ProtoData(files=XXX))`函数中指定`test.list`,例如在本例中, 测试数据已放置在正确位置，修改mnist_cluster_job.py：
  ```python
  """ 数据配置 """
  TrainData(ProtoData(files = "train.list"))
  ```

- 提交任务

```bash
python cluster_train \
  --config test/mnist/mnist_cluster_job.py \
  --use_gpu gpu \
  --time_limit 00:30:00 \
  --submitter wangyanfei01 \
  --num_nodes 2 \
  --job_priority normal \
  --trainer_count 4 \
  --num_passes 1 \
  --log_period 1000 \
  --dot_period 100 \
  --saving_period 1 \
  --where nmg01-idl-dl-cpu-10G_cluster \
  --job_name paddle_cluster_demo_v5
```
任务提交的选项已经写入`$PADDLE_PLATFORM/run_mnist.sh`中，可以直接运行：
```bash
cd $PADDLE_PLATFORM
./run_mnist.sh
```
详细解释请参阅[客户端命令行选项](#command-options)。

- 查看任务

运行提交脚本后，客户端会打印提交信息，其中包含任务的id和任务链接。在本例中，用户将看到形如`127445.nmg01-hpc-imaster01.nmg01.baidu.com`的任务id和形如`http://nmg01-hpc-controller.nmg01.baidu.com:8090/job/i-127445`的任务链接URL。打开这个页面，可以查看任务的运行状态和详细的运行日志，训练过程的日志存放在**第一个**训练节点的`workspace/log`路径下，通过`paddle_trainer.INFO`可以查看训练过程中的各个指标的变化情况。

- 查看模型

在本例中，模型在训练结束后会保存在`$fs_name/$work_dir/output_时间戳/output/rank-00000`下，包含了每个pass输出的模型。

- 杀死任务

如果在任务的运行过程中想杀死任务，可以使用客户端的`cluster_kill`子命令，在本例中：
```bash
paddle cluster_kill qdel 127445.nmg01-hpc-imaster01.nmg01.baidu.com
```
请注意不同的集群杀死任务的命令各有不同，通常在提交任务后客户端打印的提交信息中都包含了杀死任务的命令，只需要用`cluster_kill`选项将其传给客户端即可。

## AlexNet

- 准备配置文件`$PADDLE_PLATFORM/test/alexnet/alexnet_cluster_job.py`

  - 配置集群相关的参数
  ```python
  cluster_config(
          fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
          fs_ugi="paddle_demo,paddle_demo",
          work_dir="/app/idl/idl-dl/paddle/cluster_test/alexnet",
          has_meta_data=True,
          )
  ```

  - 配置训练和测试数据：
  ```python
  define_py_data_sources('train.list', 'test.list',
                         module=['pyDataProviderImage',
                                 'pyDataProviderImage_test'],
                         obj='GeneralJpegDataProvider',
                         args=['train.meta.list',
                               'test.meta.list'],
                         train_async=True)
           
  img = data_layer(name='input', size=224 * 224 * 3)
  ```
  在配置训练和测试数据时，需要用到`pyDataProviderImage`，`GeneralJpegDataProvider`，`pyDataProviderImage_test`这几个`DataProvider`，它们的实现需要用户通过客户端指定`--thirdparty`来传给trainer。

  - 配置优化算法和网络结构：
  ```python
  settings(batch_size=512,
           learning_rate=1e-3,
           learning_method=AdamOptimizer())

  tmp = img_conv_layer(input=img, filter_size=11, num_channels=3, num_filters=96,
                       stride=4, padding=1)

  tmp = img_cmrnorm_layer(input=tmp, size=5, scale=0.0048, power=0.75)

  tmp = img_pool_layer(input=tmp, pool_size=3, stride=2)

  tmp = img_conv_layer(input=tmp, filter_size=5, num_filters=256,
                       stride=1, padding=2, groups=2)

  tmp = img_cmrnorm_layer(input=tmp, size=5, scale=0.0128, power=0.75)

  tmp = img_pool_layer(input=tmp, pool_size=3, stride=2)

  tmp = img_conv_layer(input=tmp, filter_size=3, num_filters=384,
                       stride=1, padding=1)

  tmp = img_conv_layer(input=tmp, filter_size=3, num_filters=384,
                       stride=1, padding=1, groups=2)

  tmp = img_conv_layer(input=tmp, filter_size=3, num_filters=256,
                       stride=1, padding=1, groups=2)

  tmp = img_pool_layer(input=tmp, pool_size=3, stride=2)

  tmp = fc_layer(input=tmp, size=4096, act=ReluActivation())
  tmp = fc_layer(input=tmp, size=4096, act=ReluActivation())
  prediction = fc_layer(input=tmp, size=1000, act=SoftmaxActivation())

  classification_cost(input=prediction,
                      label=data_layer('label', 1000),
                      name='cost')
  Inputs('label', 'input')
  Outputs('cost')
  ```
  `paddle.trainer_config_helpers`模块提供了众多helper来简化模型网络的描述，在使用这些helper之前请先通过`from paddle.trainer_config_helpers import *`来引入该模块。
  注意到配置文件中用到了一些第三方依赖，主要是一些data provider，以及它们调用的一些动态链接库，用来处理图像的输入，所以需要将这些依赖的代码和文件以第三方依赖的形式提交，并通过客户端`--thirdparty`选项指定。

  AlexNet用到的thirdparty包含如下内容：
  ```bash
  |-- before_hook.sh
  |-- decodeJPEG
  |-- jpeg
  |-- pyDataProviderImage.py
  `-- pyDataProviderImage_test.py
  ```
  这个例子中需要使用before_hook.sh将依赖文件从thirdparty中拷贝到当前训练节点的workspace目录下，所以所有的依赖文件都需要放入名称为`thirdparty`的目录下。

- 提交任务：

```python
python cluster_train \
  --config test/alexnet/job_config/alexnet_cluster_job.py \
  --use_gpu gpu \
  --time_limit 00:05:00 \
  --submitter liuyuan04 \
  --num_nodes 1 \
  --job_priority normal \
  --trainer_count 4 \
  --num_passes 1 \
  --log_period 100 \
  --dot_period 10 \
  --saving_period 1 \
  --where cp01-idl-dl-gpu-56G_cluster \
  --job_name paddle_cluster_test_alexnet_v5 \
  --thirdparty test/alexnet/thirdparty
```
当前版本中已经包含了上述示例的所有代码，放置在`$PADDLE_PLATFORM/test/alexnet`下，通过`$PADDLE_PLATFORM/run_alex.sh`即可提交。
