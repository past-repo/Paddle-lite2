# 高级使用(Advanced Usage)

## 使用自定义PaddlePaddle

### 编译自己的PaddlePaddle
- 根据自己的需求，更新PaddlePaddle源代码，然后进入platform2，运行sh build.sh，当前
  目录下会生成一个build目录。 默认包含了cpu/gpu/rdma多种混合集群的二进制程序:
  ```bash
  ./paddle_cmd_gcc48_avx_float_api_1167/train/core/common/core_output/bin
  |-- cpu_nonrdma
  |   |-- paddle_dserver
  |   |-- paddle_pserver2
  |   `-- paddle_trainer
  |-- cpu_rdma
  |   |-- paddle_dserver
  |   |-- paddle_pserver2
  |   `-- paddle_trainer
  |-- gpu_nonrdma
  |   |-- paddle_dserver
  |   |-- paddle_pserver2
  |   `-- paddle_trainer
  `-- gpu_rdma
      |-- paddle_dserver
      |-- paddle_pserver2
      `-- paddle_trainer
  ```
  同时，build目录下还会生成编译好的二进制的依赖环境:
  ```bash
  ./paddle_cmd_gcc48_avx_float_api_1167/train/core/common/core_output/pylib/
  |-- google
  |   |-- __init__.py
  |   `-- protobuf
  `-- paddle
      |-- __init__.py
      |-- internals
      |-- proto
      |-- trainer
      |-- trainer_config_helpers
      `-- utils
  ```
- 建立目录thirdparty，根据需要使用的集群类型，将对应集群类型下的二进制文件
  paddle_trainer和paddle_pserver2，以及依赖目录google和paddle拷贝到
  thirdparty目录，并在提交任务时使用`--thirdparty=/path/to/thirdparty`选项。
- 使用before_hook.sh在训练开始前替换二进制文件和依赖环境：
  ```bash
  #!/bin/sh
  rm ./google
  rm ./paddle
  # 使用自定义paddle bin与依赖（下面是参考代码）
  mv thirdparty/thirdparty/paddle_trainer ./
  mv thirdparty/thirdparty/paddle_pserver2 ./
  mv thirdparty/thirdparty/google  ./
  mv thirdparty/thirdparty/paddle ./
  ```

