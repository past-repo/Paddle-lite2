# FAQ和附录

## FAQ
- **问**：为什么输出很多`if [ “0” -eq “0”]; `等异常日志？<br>
  **答**：这是shell 打开了 set -x debug日志，不是异常。
- **问**：有时在配置文件里使用`from something import *`之类的语法，出现cluster_config解析失败？<br>
  **答**：`import *`操作会污染trainer_config.conf全局空间，这跟receiver在获取cluster_config参数时重写了config_parser.py的cluster_config函数冲突，导致该函数置空，最终解析失败。因此不建议使用这种方法。
- **问**：cluster_config设置注意事项？<br>
  **答**：1）尽量将该函数调用放到对train.list、test.list、train.meta.list、test.meta.list的处理之前；
          2）导入trainer_config_helper的模块时避免采用`import *`,  会间接引入config_parser.py全局域。
- **问**：如何查看指定receiver支持的集群列表？<br>
  **答**：1）可以随便指定一个`--where xx`，后端receiver会因为xx是非法集群而列出所有集群列表；
          2）可以通过`-c "ls ../../train/core/nodes.configs"`在后端receiver执行获取一个列表。

## 附录(Appendix)
### 可用代理服务器(Receiver Lists)

```text
receivers=[
        "yq01-idl-gpu-offline41.yq01.baidu.com:9090"~"yq01-idl-gpu-offline41.yq01.baidu.com:9099",
        "yq01-idl-gpu-offline41.yq01.baidu.com:9190"~"yq01-idl-gpu-offline41.yq01.baidu.com:9199",
        "yq01-idl-gpu-offline41.yq01.baidu.com:9290"~"yq01-idl-gpu-offline41.yq01.baidu.com:9299",
]
```

### 可用集群(Cluster Lists)

```text
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
