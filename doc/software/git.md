# Git 使用
## 初始化
```
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
```

## 免密提交设置
```
git config --global credential.helper store
```
这一步会在用户目录下的.gitconfig文件最后添加：

    [credential]
        helper = store

push 代码(git push), 这时会让你输入用户名和密码, 这一步输入的用户名密码会被记住,
下次再push代码时就不用输入用户名密码!这一步会在用户目录下生成文件.git-credential记录用户名密码的信息。

## 基本命令
1. 查看 config 配置
```
git config -l
```
2. 忽略文件夹权限设置
```
git config core.filemode false  // 当前版本库
git config --global core.fileMode false // 所有版本库
```