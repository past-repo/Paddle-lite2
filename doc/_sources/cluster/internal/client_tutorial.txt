# 客户端教程

- **基本原理**

新平台的基本原理是由用户将任务配置投递到任务代理服务器(Receiver)，随后Receiver将任务提交到用户指定的集群。目前Receiver由IDL统一搭建与维护，您可以参阅[可用代理服务器](../faq_appendix.html#receiver-lists)来获取一份当前可用的Receiver列表，或者查看客户端安装目录下`local_config.py`中的配置来了解当前可用的receiver。
集中的Receiver服务可以让用户省去在不同集群和不同的PaddlePaddle版本间切换的成本，同时更好的管理用户的提交历史。如果您自己部署了专用的Receiver，也可以在此文件中用其替换现有Receiver。

- **任务配置**

请参阅[集群任务配置说明](../cluster_config.html)。

- **提交任务**

准备完上述模型配置文件和参数，即可提交作业，下面是一个mnist任务的提交参数：
```bash
python cluster_train \
    --config test/mnist_cluster_job.py \
    --mode gpu \
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
- **杀死任务**

用户可以通过`cluster_kill`子命令来杀死正在执行的任务，`cluster_kill`实质上接受任何shell语句，传入`cluster_kill`的语句在receiver端执行，并将结果返回给用户。具体杀死job的命令各个集群有所不同，通常提交任务后客户端的提示信息中会有完整的杀死任务的命令，将此命令作为`cluster_kill`的参数即可，例如:
对于集群`nmg01-idl-dl-cpu-10G_cluster`
```bash
paddle cluster_kill qdel $your_job_id $param1 $param2 ...
```
对于集群`yq01-idl-idl-offline_slurm_cluster`
```bash
paddle cluster_kill deljob $your_jobid $param1 $param2 ...
```

- **查看模型**

在训练期间，用户可以通过`$node_ip:$model_http_server_port`获取模型和训练日志，其中`$node_ip`为训练时分配的计算节点的ip，通常模型存储在主节点上，日志文件在各个节点上都存在。 默认的`$model_http_server_port`为`8099`。例如：
获取某一轮的模型:
```bash
wget -r -np -nH --cut-dirs=1 -R index.html 10.73.213.33:8099/output/pass-00000
```
获取训练日志:
```bash
wget 10.73.213.33:8099/log/paddle_trainer.INFO
```
在训练结束后，模型和日志会上传到HDFS上，存储在`$output_path/output/rank-00000`，日志存储在`$output_path/log/rank-xxxxx`。

- **查看当前二进制版本**

用户在输出的`模型文件夹`中会存有版本文件。

- **自定义Receiver**

如须自定义Receiver，请修改`local_config.py`中的`receivers`变量，用自定义receiver替换原有receiver列表。

- **获取支持的MPI集群列表**

常用的技巧，可以获取选定的receiver支持的MPI集群列表，通过指定一个错误--where参数，后端receiver会吐出所有MPI集群列表
比如使用 `--where xx`会返回以下列表
```
yq01-idl-idl-offline_slurm_cluster
yq01-idl-idl-offline_vfs_slurm_cluster
yq01-idl-idl-offline_vfs_metric_learning_slurm_cluster
yq01-ecom-triger-box_slurm_cluster
yq01-msbu-dpp-offline_slurm_cluster
cp01-idl-dl-gpu-56G_cluster
nmg01-idl-dl-cpu-10G_cluster
nmg01-hpc-off-dmop-cpu-10G_cluster
nmg01-hpc-off-dmop-slow-cpu-10G_cluster
```
