# 客户端安装

* 安装jumbo：
```bash
bash -c “$( curl http://jumbo.baidu.com/install_jumbo.sh )”
source ~/.bashrc
```
* 更新jumbo源并安装客户端
```bash
jumbo add_repo http://m1-idl-gpu2-bak31.m1.baidu.com:8088/jumbo/alpha/
jumbo install paddle_platform
```
* 运行客户端
当前的客户端安装在`$JUMBO_ROOT/opt/paddle_platform`下，下文中将用`$PADDLE_PLATFORM`来代指平台的安装目录。客户端的详细使用方法参见[客户端教程](../client_tutorial.html)。
```bash
paddle cluster_train -h
```

除客户端之外，新平台还包含服务端，用户通过客户端提交的配置会先抵达服务端，再由服务端提交到用户指定的运行环境中。通常情况下，您不需要自己部署服务端。如果您想了解服务端部署，请参阅[服务端部署](../server_deployment.html)。
