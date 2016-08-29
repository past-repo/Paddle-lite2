# 集群任务配置说明

## 集群配置文件(Config File)

集群配置的相关选项需要通过调用`cluster_config`方法来指定。**注意**：如果配置文件需要引入任何第三方依赖模块`import $some_thirdparty`，请将`cluster_config`放置在`import $some_thirdparty`前面否则，可能会调用失败。

具体的接口定义如下：

```python
cluster_config(
    fs_name = None,
    fs_ugi = None,
    force_reuse_output_path = False,
    work_dir = None,
    has_meta_data = False,
    train_data_path = None,
    test_data_path = None,
    train_meta_data_path = None,
    test_meta_data_path = None,
    output_path = None,
    init_model_path = None,
    pserver_model_dir = None,
    pserver_model_pass = None,
    save_dir = "output",
    loadsave_parameters_in_pserver = False,
    enable_predict_output = False,
    model_path = None,
    port = 7164,
    ports_num = 1,
    use_remote_sparse = False,
    mail_address = None,
    comment = None)
```
- **fs_name**:
  - 当前分布式训练任务数据及模型存放的HDFS名称
  - 示例：`fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310"`
- **fs_ugi**:
  - 当前分布式训练任务用到的HDFS的UGI
  - 示例：`"paddle_demo,paddle_demo"`
- **force_reuse_output_path**: 
  - 如果赋值为true, 会直接移除`output_path`而不检查其是否存在
- **checking output_path exist**: 
  - 检查指定的输出目录是否存在，默认值false。
- **work_dir**：
  - 如果指定了`work_dir`，会按照如下方式指定：
    - train_data_path = work_dir + "/train"
    - test_data_path = work_dir + "/test"
    - train_meta_path = work_dir + "/train_meta_dir" (如果使用meta_data)
    - test_meta_path = work_dir + "/test_meta_dir" (如果使用meta_data)
    - output = work_dir + "/output"
- **has_meta_data**: 
  - 默认值false, 如果设置为true, 自动生成`*_meta_path`, 否则无行为
  - 如果指定了`work_dir`并且存在`meta_data`，请将本选项设置为true
- **train_data_path**: 
  - 训练数据在HDFS上的路径
  - 示例：`train_data_path="/app/idl/idl-dl/paddle/demo/mnist/train"`
- **test_data_path** : 
  - 测试数据在HDFS上的路径
  - 如`test_data_path="/app/idl/idl-dl/paddle/demo/mnist/test"`
- **train_meta_data_path**: 
  - 训练数据元数据路径，默认为None。
- **test_meta_data_path**: 
  - 测试元数据路径，默认为None。
- **output_path**: 
  - 输出文件在HDFS上的路径
  - 示例：`output_path="/app/idl/idl-dl/paddle/demo/mnist/output_\`date +%Y%m%d%H%M%S\`"`
- **model_path**: 
  - 模型在HDFS上的保存路径，模型评估时`tester`会从该路径读取模型文件，默认值是None
  - 示例：`model_path="output"`
-- **init_model_path**: 
    指指定初始化模型路径。如果需要从已有模型开始训练，请指定该参数
  - 其他情况下不需要指定该参数，如`init_model_path=""`
- **pserver_model_dir**:
  - 为pserver指定的初始化模型路径
- **pserver_model_pass**:
  - 示例：如果设置`pserver_model_dir = "/app/paddle/models`和`pserver_model_dir = "/app/paddle/models`
    - rank0将从`/app/paddle/models/rank-00000/pass-00123`下载模型
    - rank1将从`/app/paddle/models/rank-00001/pass-00123`下载模型
    - 依次类推
- **save_dir**:
  - paddle_trainer的模型存放路径，如果指定的值不是`output`的话，模型就不会被上传到HDFS
  - 示例：`save_dir="output"`
- **loadsave_parameters_in_pserver**:
  - 是否由pserver来保存模型
  - 0表示由paddle_trainer来保存模型，1表示由pserver来保存模型
- **enable_predict_output**:
  - 在评估模式下，如果设置为1，用户可以获取不同的Layer的值，缺省为0
- **port**:
  - pserver端口
- **port_num**:
  - pserver链接的端口数量
  - 在使用了fat channels的情况下，增加该数量可以更好的发挥集群性能
- **use_remote_sparse**:
  - 是否使用远程稀疏更新
  - 示例：`use_remote_sparse = True`
- **comment**:
  - 通过此选项为pserver和trainer添加注释
  - 示例：`comment="${PBS_O_LOGNAME}_${PBS_JOBID}"`
- **mail_address**:
  - 如果存在慢节点，可以通过在此设定的邮箱发送报警
  - 示例：`mail_address="example_user@baidu.com"`

## 命令行选项(Command Options)

客户端除需要指定配置文件之外，主要包括任务相关的配置，例如任务名称，集群类型，提交者邮箱等。具体的配置选项如下：
```bash
submit job to general scheduler center.

optional arguments:
  -h, --help            打印帮助信息
  -w WHERE, --where WHERE
                        物理集群名称
  -n NUM_NODES, --num_nodes NUM_NODES
                        训练使用节点数量
  -j JOB_NAME, --job_name JOB_NAME
                        作业名称
  -s SUBMITTER, --submitter SUBMITTER
                        作业提交者邮箱前缀
  -l TIME_LIMIT, --time_limit TIME_LIMIT
                        作业最长运行时间
  -p JOB_PRIORITY, --job_priority JOB_PRIORITY
                        作业优先级
  -t USE_GPU, --use_gpu USE_GPU
                        训练类型，默认为"gpu"，如使用CPU，请指定"cpu"
  -f CONFIG, --config CONFIG
                        任务的配置文件
  -i THIRDPARTY, --thirdparty THIRDPARTY
                        任务的第三方依赖文件所在目录
  -c SHELL, --shell SHELL
                        自定义shell命令，此命令将被转发到Receiver服务器上执行
```
除以上选项外，所有可被`paddle train`接受的参数都可以通过客户端指定，这些参数会通过客户端传递给各个训练节点上的trainer进程，例如：
```bash
  --trainer_count 4 \
  --num_passes 1 \
  --log_period 1000 \
  --dot_period 100 \
  --saving_period 1
```
新平台后台对参数会做部分CHECK和优化动作，如会自动重写trainer_count = 16以最大化box集群GPU计算资源。其他参数的合法性需要多机主进程trainer运行时CHECK。

## 第三方依赖(Thirdparty Dependencies)

`--thirdparty`可以看作一个开放式接口。在某些情况下，用户的配置文件需要依赖第三方代码和库。例如输入图像时需要专门的data provider和一些第三方的动态库，这些文件可以统一放置在一个目录下，通过客户端的`--thirdparty`选项上传。通常情况下，平台会将`thirdparty`加入到paddle train主进程的python环境路径，用户在配置文件里直接引用即可。另一方面，用户可以通过`before_hook.sh`对训练环境进行精细化的自定义，以满足训练时的各种自定义需求, 例如使用自定义版本的`paddle_trainer`和`paddle_pserver`等，都可以通过`--thirdparty`配合`before_hook.sh`实现。

假定通过客户端指定`--thirdparty $user_path/$your_thirdpary_dir`，`$user_path/$your_thirdparty_dir`中的文件在集群节点上将会放置在`./thirdparty/$your_thirdparty_dir`下。集群python运行时环境：

```bash
PYTHONPATH=paddle:./thidparty/$user_path/thirdpary
PYTHONHOME=./python27-gcc482
```

其中paddle模块的目录结构如下：
```bash
paddle
    |-- __init__.py
    |-- __init__.pyc
    |-- internals
    |-- proto
    |-- trainer
    |-- trainer_config_helpers
    `-- utils
```

- gcc48运行时环境的python解释器

用户可以单独下载供单机测试：

```bash
hadoop dfs -Dfs.default.name=hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 -Dhadoop.job.ugi=paddle_demo,paddle_demo -ls /app/idl/idl-dl/paddle/tools/python27-gcc482.tar.gz
```

- before_hook后门执行脚本

用户可以在`$user_path/thirdpary`目录下添加自定义脚本before_hook.sh。通常情况下这个脚本可以用来将依赖文件根据线上运行时环境进行部署，但实际上不仅限于此，任何可以运行的shell脚本都可以作为before_hook.sh使用。

before_hook.sh**正常执行须满足如下条件**：1）必须放置在`$user_path/thirdpary`目录下；2）目录的名称必须为`thirdparty`。
这样每个训练节点上trainer启动前，都会先执行before_hook.sh。
