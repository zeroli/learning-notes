整体看下sstable的组要组成，如下：
===
![](images/2022-02-14-20-28-19.png)

sstable生成细节
===
sstable生成时机：
* minor compaction
  immutable-memtable中的key/value dump到磁盘，生成sstable
* major compaction
  sstable compact(level-n sstable(s)与level-n+1 sstable多路归并)，生成level-n+1的sstable

首先是写入data block:
===
![](images/2022-02-15-05-36-16.png)

data block都写入完成后，接下来是meta block:
===
![](images/2022-02-15-05-36-39.png)

然后是data/meta block索引信息data/meta index block写入:
===
![](images/2022-02-15-05-36-57.png)

最后将index block的索引信息写入Footer
===
![](images/2022-02-15-05-37-13.png)

一个完整的sstable形成!
