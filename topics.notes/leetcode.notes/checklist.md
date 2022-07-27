| #index | description | status | comment |
| --- | --- | --- | --- |
| 1 | Two Sum | DONE | 采用hash table来辅助查找另一个数 |
| 2 | Add Two Numbers | | |
| 3 | Longest Substring Without Repeating Characters | | |
| 7 | Reverse Integer | DONE | 注意overflow/underflow的检测 |
| 8 | String to Integer (atoi) | DONE | 注意overflow/underflow的检测 |
| 14 | Longest Common Prefix | DONE | |
| 100 | Same Tree | DONE | 采用递归，判断当前节点之后，在分别判断左右子树；或者层序遍历；或者判断前序和中序都相同 |
| 104 | Maximum Depth of Binary Tree | DONE | 采用递归来实现，节点深度为左右深度最大值 + 1 |
| 206 | Reverse Linked List | DONE | 递归和迭代都可以 |
| 141 | Linked List Cycle | DONE | 快慢指针，检测有环 |
| 160 | Intersection of Two Linked Lists | DONE | 两个指针各自遍历，到底各自末尾时，转向另一个链表头节点，然后再遍历直到相遇，为交点(这个有问题，没有交点会卡死）。求各自链表长度，长者先走diff个，然后再一起走直到相遇或为null) |
