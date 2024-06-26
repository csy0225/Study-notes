# 一、更新系统软件包并安装必要依赖
1. 首先，更新 Ubuntu 的软件包列表
   ``` bash
   sudo apt-get update
   ```
2. 接下来，安装所需的软件包，这些软件包允许 apt 使用 https 来访问仓库：
   ``` bash
   sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
   ```
# 二、添加 Docker 官方 GPG 密钥和软件仓库
1. 导入 Docker 的官方 GPG 密钥：
   ``` bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   ```
2. 添加 Docker 软件仓库。请确保下面命令中的 $(lsb_release -cs) 能够正确输出你的 Ubuntu 发行版名称，如 focal、bionic 等；
   ``` bash
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   ```
# 三、安装 Docker 引擎
1. 再次更新软件包列表以包含新添加的 Docker 仓库
   ```bash
   sudo apt-get update
   ```
2. 安装最新版本的 Docker 引擎
   ``` bash
   sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```
# 四、启动和验证 Docker 安装
1. 启动 Docker 服务
   ```
   sudo systemctl start docker
   ```
2. 验证 Docker 服务是否在运行
   ```
   sudo systemctl status docker
   ```
3. 检查 Docker 版本以确认安装成功：
   ```
   docker --version
   ```
# 五、配置用户权限（可选）
1. 默认情况下，只有 root 用户或具有 sudo 权限的用户可以运行 docker 命令。若要以非 root 用户身份运行 docker，可以将用户添加到 docker 组中
   ```
   sudo usermod -aG docker ${USER}
   ```
2. 添加用户到 docker 组后，需要重新登陆才能使更改生效。



总结：完成以上步骤后，你就应该在 Ubuntu 上成功安装了 Docker。如果需要进一步验证安装，可以尝试运行一个简单的 Docker 容器，例如：
```
docker run hello-world
```
