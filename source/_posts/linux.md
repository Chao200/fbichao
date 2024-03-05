# 基本命令

- pwd (*print working directory*)

    当前文件目录

- ls (*list*)

    列出当前目录下的内容

    - ls `-a`：列出隐藏文件
    - ls `-l /`：列出根目录下文件的详细信息
    - ls `-l -h /`：更易阅读的方式显示

- cd(*change directory*)
    - cd `..`：返回上级目录
    - cd `~`：根目录
    - cd `/`：绝对路径
    - cd `./`：相对路径
    - cd `-`：返回之前目录


- cat
    
    输出文件内容

- head / tail
    - head/tail --lines=3 file：显示 file 前/后三行

- less / more

    显示文件完整内容


- nano

- vim

- file

    显示文件属性

- where

    查找程序位置


- echo

    打印

    ```zsh
    h="hello"
    echo $h
    echo "${h} world"
    ```

- mv

    重命名，移动

- for

    把目录下所有 week 开头的改成 chapter 开头
    ```zsh
    for ff in week*
    do
    mv ff chapter${ff#week}
    done
    ```





- 命令 `-h` 和 `--help`
- `man` 命令

