# 更新列表(Change Log)

## 内测后端

为了保证后端服务的稳定可靠，新特性会首先升级到内测Receiver供感兴趣的用户试用，
内测稳定后再同步到Production Receiver(客户端默认Receiver列表)。内测Receiver列表
如下：
```python
# upgraded to latest with timer
"yq01-idl-gpu-offline41.yq01.baidu.com:7890",
"yq01-idl-gpu-offline41.yq01.baidu.com:7891",
"yq01-idl-gpu-offline41.yq01.baidu.com:7892",
"yq01-idl-gpu-offline41.yq01.baidu.com:7893",
"yq01-idl-gpu-offline41.yq01.baidu.com:7894",
"yq01-idl-gpu-offline41.yq01.baidu.com:7895",
"yq01-idl-gpu-offline41.yq01.baidu.com:7896",
"yq01-idl-gpu-offline41.yq01.baidu.com:7897",
"yq01-idl-gpu-offline41.yq01.baidu.com:7898",
"yq01-idl-gpu-offline41.yq01.baidu.com:7899",
# upgraded to latest without timer
"yq01-idl-gpu-offline41.yq01.baidu.com:7990",
"yq01-idl-gpu-offline41.yq01.baidu.com:7991",
"yq01-idl-gpu-offline41.yq01.baidu.com:7992",
"yq01-idl-gpu-offline41.yq01.baidu.com:7993",
"yq01-idl-gpu-offline41.yq01.baidu.com:7994",
"yq01-idl-gpu-offline41.yq01.baidu.com:7995",
"yq01-idl-gpu-offline41.yq01.baidu.com:7996",
"yq01-idl-gpu-offline41.yq01.baidu.com:7997",
"yq01-idl-gpu-offline41.yq01.baidu.com:7998",
"yq01-idl-gpu-offline41.yq01.baidu.com:7999",
"yq01-idl-gpu-offline41.yq01.baidu.com:9990",
"yq01-idl-gpu-offline41.yq01.baidu.com:9991",
"yq01-idl-gpu-offline41.yq01.baidu.com:9992",
"yq01-idl-gpu-offline41.yq01.baidu.com:9993",
"yq01-idl-gpu-offline41.yq01.baidu.com:9994",
"yq01-idl-gpu-offline41.yq01.baidu.com:9995",
"yq01-idl-gpu-offline41.yq01.baidu.com:9996",
"yq01-idl-gpu-offline41.yq01.baidu.com:9997",
"yq01-idl-gpu-offline41.yq01.baidu.com:9998",
"yq01-idl-gpu-offline41.yq01.baidu.com:9999",
```

## 后端更新
- 支持cuda、cudnn 动态加载，支持单一receiver能统一调度所有cpu和所有gpu集群，包括
  normandy mpi和slurm mpi集群。
- 支持对小节点作业自动优化port参数，提高网络并行度和pserver计算cpu并行度。
- 支持test为为空。
- 支持运行时通过特定端口wget 模型和日志。
- 修复matrix基础平台升级导致的网络接口信息获取失败的bug。
- 支持模型里获取paddle版本号。
- 后端已接入yq01-msbu-dpp-offline_slurm_cluster、
  yq01-ecom-triger-box_slurm_cluster、nmg01-hpc-off-dmop-cpu-10G_cluster第三方集
  群集群。
- fix bug：恢复对local=1重写功能。
- fix bug：train和test混合设置导致数据错误的bug。
- monitor模块支持对历史作业统计并定时发送摘要邮件，方便leader了解后端服务情况。
- `upgraded to latest with timer`的Receiver支持获取详细的分布式训练效率，可以帮
  助有效定位多机训练的瓶颈。

## 客户端更新

### v5.11.alpha-1
- 支持paddle cluster_kill子命令杀死作业。

