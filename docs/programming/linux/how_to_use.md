# linux-command-line

## information about system | applications
- 查看ubuntu版本 sudo lsb_release -a
- free -m 查看内存使用情况
- 查看驱动版本：cat /proc/driver/nvidia/version
- ps
    - ps aux|grep lk|grep python|awk '{print $2}'|xargs kill
    - ps -ef | grep firefox   pgrep firefox
    - ps -ef | grep firefox | awk '{print $2}' | xargs kill -9
        - 其中awk '{print $2}' 的作用就是打印（print）出第二列的内容。根据常规篇，可以知道ps输出的第二列正好是PID。就把进程相应的PID通过xargs传递给kill作
参数，杀掉对应的进程。

## file system
- 压缩解压缩命令
    - tar cjf - logs/ |split -b 1m - logs.tar.bz2. 将logs文件分包压缩成1m大小的切片文件
    - tar cjf - install/ |split -b 260m - tuzhenalg2.0.2.tar.bz2.
    - tar -zcvf sdk.tar.gz ./sdk
    - tar -zcvf tuzhensdk2.1.x.tar.gz ./install
- 文件查找
    - sudo find / -name libjson_linux-gcc-4.6_libmt.so 查找so文件所在的路径
    - whereis nano
    - which nano
    - find -L . -type l  查找当前目录下失效的软链接
- scp file usr@ip:/home/user/path 拷贝本地文件到远程文件
- 磁盘信息
    - df -hl 查看磁盘空间
    - du -sh 查看当前文件夹大小
    - du -sh * | sort -n 统计当前文件夹(目录)大小，并按文件大小排序
    - du -sk filename 查看指定文件大小
    - du -h --max-depth=1
- link
    - 硬连接
        - 硬连接指通过索引节点来进行连接。在Linux的文件系统中，保存在磁盘分区中的文件不管是什么类型都给它分配一个编号，称为索引节点号(Inode Index)。在Linux中，
        多个文件名指向同一索引节点是存在的。一般这种连接就是硬连接。硬连接的作用是允许一个文件拥有多个有效路径名，这样用户就可以建立硬连接到重要文件，
        以防止“误删”的功能。其原因如上所述，因为对应该目录的索引节点有一个以上的连接。只删除一个连接并不影响索引节点本身和其它的连接，只有当最后一个连接被删除后，
        文件的数据块及目录的连接才会被释放。也就是说，文件真正删除的条件是与之相关的所有硬连接文件均被删除。
    - 软连接
        - 符号连接（Symbolic Link），也叫软连接。软链接文件有类似于Windows的快捷方式。它实际上是一个特殊的文件。在符号连接中，文件实际上是
        - ln -s source dest
- cat主要有三大功能：
    - 1.一次显示整个文件:cat filename
    - 2.从键盘创建一个文件:cat > filename 只能创建新文件,不能编辑已有文件.
    - 3.将几个文件合并为一个文件:cat file1 file2 > file 一个文本文件，其中包含的有另一文件的位置信息。

## dynamic link 
- ldd是list, dynamic, dependencies的缩写， 意思是， 列出动态库依赖关系,如：ldd libcaffe.so | grep proto 列车当前文件库下的动态文件;
- ldconfig几个需要注意的地方 
    - 1. 往/lib和/usr/lib里面加东西，是不用修改/etc/ld.so.conf的，但是完了之后要调一下ldconfig，不然这个library会找不到 
    - 2. 想往上面两个目录以外加东西的时候，一定要修改/etc/ld.so.conf，然后再调用ldconfig，不然也会找不到,比如安装了一个mysql到/usr/local/mysql，mysql有一大堆library在/usr/local/mysql/lib下面，这时 就需要在/etc/ld.so.conf下面加一行/usr/local/mysql/lib，保存过后ldconfig一下，新的library才能在 程序运行时被找到。 
    - 3. 如果想在这两个目录以外放lib，但是又不想在/etc/ld.so.conf中加东西（或者是没有权限加东西）。那也可以，就是export一个全局变 量LD_LIBRARY_PATH，然后运行程序的时候就会去这个目录中找library。一般来讲这只是一种临时的解决方案，在没有权限或临时需要的时 候使用。 
    - 4. ldconfig做的这些东西都与运行程序时有关，跟编译时一点关系都没有。编译的时候还是该加-L就得加，不要混淆了。 
    - 5. 总之，就是不管做了什么关于library的变动后，最好都ldconfig一下，不然会出现一些意想不到的结果。不会花太多的时间，但是会省很多的事。

