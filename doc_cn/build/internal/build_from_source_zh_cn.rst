使用CMake编译PaddlePaddle源码
==========================

.. contents:: 目录

PaddlePaddle支持COMAKE和cmake两种方式编译。由于comake本身在公司并不继续维护，
并且comake并不适合PaddlePaddle对外开放使用，所以PaddlePaddle目前优先支持cmake编译
comake编译目前
进入 **deprecated** 状态，并会在未来的数月内删除。

CMake编译的方法主要分为一下几个步骤执行:

* 下载编译依赖
* 设置编译选项，生成Makefile
* 编译Makefile，安装

用户可以选择左侧的目录跳着阅读感兴趣的模块。对于CMake并不了解的同学，可以参考文档
`cmake简介 <https://cmake.org/cmake-tutorial/>`_

在编译之前，请 `下载PaddlePaddle源码 <./download_paddle_source_zh_cn.html>`_


下载编译依赖
--------------

下载编译依赖目前可以使用 Jumbo_ 和百度SVN下载。下面的方式二选一即可

Jumbo_ 下载编译依赖
````````````````````

使用 Jumbo_ 下载PaddlePaddle的依赖非常简单。分为:

*  安装 Jumbo_ 。
*  执行PaddlePaddle编译依赖安装脚本。


安装 Jumbo_
..........................


..  code-block:: bash

    bash -c "$( curl http://jumbo.baidu.com/install_jumbo.sh )"
    source ~/.bashrc


然后执行


..  code-block:: bash

    jumbo



即可以检测jumbo是否安装完成。

执行PaddlePaddle编译依赖安装脚本
.............................

执行目录 :code:`paddle/internals/scripts/build_scripts/install_deps.jumbo.sh` 这个shell脚本。即可完成PaddlePaddle编译环境的依赖安装。 脚本内容如下\:

..  include:: ../../paddle/internals/scripts/build_scripts/install_deps.jumbo.sh
    :code: bash
    :literal:

其中使用jumbo安装了 :code:`protobuf`, :code:`gflags`, :code:`glogs` 等PaddlePaddle需要使用的C++库，和若干PaddlePaddle需要使用的python库。

百度SVN下载编译依赖
````````````````````

To Be Defined




设置编译选项，生成Makefile
-------------------------------

这一步分为以下几个步骤:

* 选择编译路径
* 设置CMake编译选项


选择编译路径
``````````````````

理论上CMake可以使用任何路径作为编译路径，为了接下来教程写起来更方便，这里指定 :code:`YOUR_CODE_CHECKOUT_DIR/build` 为编译路径。 执行如下命令，创建编译路径\:

..  code-block:: bash

    cd YOUR_CODE_CHECKOUT_DIR  # 修改成代码下载地址！
    mkdir -p build
    cd build

设置CMake编译选项和GCC路径
``````````````````````````````````

编译PaddlePaddle需要使用至少gcc 4.6版本的c++。gcc 4.6到gcc 4.9测试过均可编译。gcc 5+没有经过测试。

使用公司统一部署的gcc482，需要修改PATH，执行

..  code-block:: bash

    export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH

如果使用jumbo里的gcc46，则需要修改PATH，执行

..  code-block:: bash

    export PATH=${JUMBO_ROOT}/opt/gcc46/bin:$PATH


设置CMake编译选项有两个方式。直接执行 :code:`cmake` 命令或者使用cmake命令的图形化界面 :code:`ccmake`。使用 :code:`-D` 可以将设置传递给 :code:`cmake`.


目前常用的CMake编译选项包括, 如果不设置这些编译选项，PaddlePaddle会选择一个比较好的默认值(例如在GPU的机器上开启GPU编译)

..  include:: ../../CMakeLists.txt
    :start-line: 22
    :end-line: 36
    :code: cmake

同时，CMake的一些常用的搜索路径和其他设置包括:

* CUDNN_ROOT。 CUDNN的搜索路径。需要设置成jumbo下载后的结果。
* CMAKE_INSTALL_PREFIX。编译后的安装路径。
* CMAKE_BUILD_TYPE。 是否Debug编译。包括 Debug, Release, RelWithDebInfo, MinSizeRel。


一个典型的CMake编译选项设置调用如下\:

..  code-block:: bash

    cmake -DCMAKE_INSTALL_PREFIX="${JUMBO_ROOT}" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCUDNN_ROOT="${JUMBO_ROOT}/opt/cudnn/" \
        -DWITH_SWIG_PY=ON ..

执行完上述代码后，如果正常退出没有报错，CMake配置完毕，已经生成了Makefile。


编译Makefile，安装
----------------------

执行编译命令\:

..  code-block:: bash

    make >/dev/null

如果用户如果不关注编译过程，可以将stdout重定向到:code:`/dev/null`。

编译正确以后，可以执行单元测试，验证PaddlePaddle运行正确(可选项)。


..  code-block:: bash

    export LD_LIBRARY_PATH=/home/test_jumbo/.jumbo/opt/  # 添加CUDNN到$LD_LIBRARY_PATH中
    make test

如果单元测试完全正确了，即可以安装PaddlePaddle。执行

..  code-block:: bash

    make install

注意：运行PaddlePaddle的各项命令之前，需要确保cudnn的动态链接库在$LD_LIBRARY_PATH中。

.. _Jumbo: http://jumbo.baidu.com/
