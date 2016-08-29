# 服务端部署
如果您确定自己所使用的集群已部署好PaddlePaddle新平台，可以跳过此章节。如果您想让自己使用的集群能够通过新平台提交任务，通常不需要自己搭建服务端，您只需要将集群的具体信息(包括服务器的地址、集群类型、软件环境等)提供给我们，我们在现有的Receiver中添加上该集群对应的配置文件，就可以正常使用了。

PaddlePaddle新平台的服务端部署主要指用户搭建一个或多个```Receiver```（您可以将Receiver看做是一个任务代理服务器，您提交的任务配置和其他文件首先被Receiver接收，随后Receiver根据配置再将任务提交的指定的集群），多个Receiver可以搭建在同一台机器上，同时部署一个```log_server```来收集Receiver日志，一个```monitor```来监控Receiver的运行状态。

新平台的服务端架构如下图所示：

![platform_arch](paddle_platform_arch.png)

## Receiver部署
- 下载源码
- 编译打包平台化bin包：部分提交集群需要特殊的运行时环境，比如box集群，所以建议特殊集群自己编译打包。
   - 按照paddle编译流程，下载完整的源码包
   - 进入platform2目录，`sh build.sh`，在当前目录会生产一个build输出。 默认包含了cpu/gpu/rdma多种混合集群的二进制程序
- 更新系统环境变量
```bash
cd platform2/scheduler/receiver/tools
sh install_hpc_client.sh
```
- 设置自动部署脚本

因为平台化脚本提交过程包含很多本地磁盘文件的修改过程，同时每个receiver需要独立维护一个平台化脚本，所以默认不支持同时提交多个任务。现在部署方法简单粗暴，支持克隆一份完整的平台化和scheduler脚本，实现一个完整的receiver instance。
`platform2/scheduler/receiver/tools`提供了一个自动部署脚本，会根据配置自动构造若干receiver的所需环境，需要设置一下参数：

```python
receivers = [
    "127.0.0.1:9090",
    "127.0.0.1:9091",
    "127.0.0.1:9092",
    ]
platform = "/home/wangyanfei/paddle.ci.platform/idl/paddle/platform2.next/build/paddle_cmd_gcc48_avx_float_api_1052/train"
scheduler = "/home/wangyanfei/paddle.ci.platform/idl/paddle/platform2.next/scheduler"
deploy_dir = "/home/wangyanfei/tmp/platform2_deployment"
log_server = "yq01-idl-gpu-offline14.yq01.baidu.com:9900"
```
  1. receivers： 启动server端口和监听地址
  2. platform：  本机的platfrom客户端目录
  3. scheduler：调度模块的目录
  4. deploy_dir:  部署目录
  5. log_server:  日志服务地址，receiver会将相关job请求发送到log服务

脚本会在`deploy_dir`克隆若干副本，并启动相关scheduler服务。

- 修改支持的集群列表

部分receiver可能不支持某些特殊集群，比如box集群因为环境特殊，需要在box集群编译的PaddlePaddle二进制，所以此时可以disable对应的集群配置，达到禁止通过该receiver提交特定集群的功能。
比如，对于非box节点上的receiver禁止调度执行box集群的作业，那么直接删除平台化目录内的`paddle_cmd_gcc48_avx_float_api_1083/train/core/nodes.configs/yq01-ecom-triger-box_slurm_cluster`文件夹即可。

- 其他

部署脚本生成的receiver instances目录，文件名以receiver地址和端口号命名。建议receiver使用hostname形式的receiver地址而不是ip地址形式，提高可读性，容易确定receiver部署在什么机房。

## Log Server部署
Log Server用于收集一个或多个Receiver产生的日志，当前记录的内容包括```Job Config, Tracking Url```等信息，这些信息在未来可以用于详细的实验历史分析，帮助用户分析参数调整的过程和效果的变化。在前面的Receiver部署中，用户已经指定了Log Server的地址，这个地址可以与Receiver在相同的机器上也可以使用单独的机器。如果要开启Log Server服务，请将Log Server启动脚本拷贝到您部署Receiver时指定的log_server机器上，启动脚本位于您部署Receiver时指定的```deploy_dir/receiver_name/scheduler/receiver/log_server```下面。

启动Log Server的命令如下:
```bash
usage: log_server.py [-h] [-p PORT] [-f LOG_FILE]

log paddlepaddle jobs and signal hetu.baidu.com

optional arguments:
    -h, --help            show this help message and exit
    -p PORT, --port PORT  port of logging server
    -f LOG_FILE, --log_file LOG_FILE
                          running log for log server
```
此处须注意指定的端口，要与部署Receiver时指定的log_server端口一致。

## Monitor部署
Monitor会监控receivers服务状态，间隔（默认60sec）一段时间确认receiver是否alive，并对异常、异常恢复等发送邮件通知，确保服务正常运作。
```python
"""
all receivers needed to be check alive
"""
receivers = [
        "yq01-ecom-triger-box01.yq01.baidu.com:9090",
        ...
        ]
"""
mail list to receive warning message

if any receive failed, send warning mails
send abstract mail each week even if no failure is found.
"""
warning_maillist = [
        "wangyanfei01@baidu.com",
        "luotao02@baidu.com",
        "yuyang18@baidu.com",
        "dangqingqing@baidu.com",
        ]
```
Monitor位于log_server的同级目录。您只需要配置被监控的receivers的列表，以及报警的通知邮件列表，然后启动monitor服务：
```bash
python monitor.py
```
## 接入新MPI集群
### 接入物理集群(Physical Cluster)
新平台化架构在理论上对于任意一个receiver允许接入任意的MPI集群，包括接入GPU集群和CPU集群。
实际上，不同物理集群表现的特殊性覆盖层面较多，比如高速互联网路底层配置和RDMA相关运行时环境相应接口，GPU运行时环境、物理GPU卡数目和cudnn运行时库版本支持，物理集群对应调度器版本和调度器运行时依赖等等。
甚至例如BOX集群完全非标准化服务器等，会导致极少量集群在支持新MPI集群的时候需要重新部署单独的receiver。

一般情况下，为了支持接入一个新的物理MPI集群，仅仅需要在相应的receiver端的`train/core/nodes.configs/`下准备一个新的集群配置即可。

TODO: 示例接入一个新的CPU集群和新的物理集群

### 接入虚拟集群(Visual Cluster)
新平台为了优化用户体验，让常规用户无需关注集群相关细节。但是用户如果想高效利用集群，可能需要设置集群部分参数，如采用GPU计算和CPU端存储混部的集群架构，来满足对大数据存储和访问的高效性。
为此，新平台通过虚拟集群的设计来满足此类需求。

虚拟集群定义为`物理集群+特定集群配置`的一个集群，普通用户完全不用考虑两者的区别，仅需要关心特定集群的使用方法。
比如，yq01-idl-idl-offline_slurm_cluster、yq01-idl-idl-offline_vfs_slurm_cluster和yq01-idl-idl-offline_vfs_metric_learning_slurm_cluster，这个三个虚拟集群同属一个物理集群。
其中vfs关键字的集群混部了HDFS存储和GPU计算，对于图像训练上TB的数据，可以避免训练前等待数据下载，优化训练周期时间和数据访存时间。
