# redis-1.3.7代码阅读

这个版本的代码定义的数据结构比较少，只有下面几个：
1. sds: 字符串
2. adlist：双向链表
3. dict：字典（hashtable）
4. zipmap：string => string的map
