# 书<MySQL是怎样运行的 - 从根儿上理解MySQL> 阅读笔记

第4章： 从一条记录说起——InnoDB记录存储结构
===

4.2 InnoDB页简介
===
InnoDB采取的方式就是将数据划分为若干个页，以页作为磁盘和内存之间交互的基本单位。InnoDB中页的大小一般位16KB。
