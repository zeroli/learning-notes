插入一条K/V记录
===
![](images/2022-02-14-20-17-46.png)

持有Writer的线程进入Writers队列,细节如下：
===
![](images/2022-02-14-20-18-21.png)

MakeRoomForWrite的流程图：
===
![](images/2022-02-14-20-21-01.png)

记录会首先写入磁盘上的binlog，避免程序crash时内存数据丢失：
===
![](images/2022-02-14-20-24-37.png)

K/V记录插入内存中的Memtable:
===
![](images/2022-02-14-20-26-30.png)
