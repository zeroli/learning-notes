# slist的学习和研究

文件列表：
- stl_slist.h

### `slist`类
这个类是个模板类，继承于基类模板类`_Slist_base`，模板参数就是数据类型和内存分配器类型，与其它容器类型定义保持一致，当时它还不属于C++标准规定的容器类型。单链表容器类型。

提炼出一个node_base类：它只包含一个指向下一个节点的指针。
```CPP
struct _Slist_node_base
{
  _Slist_node_base* _M_next;
};
```
然后node类继承于这个基类：
```CPP
template <class _Tp>
struct _Slist_node : public _Slist_node_base
{
  _Tp _M_data;
};
```
**为啥要这样设计？**
这样设计的原因在于slist设计有一个哨兵头节点，它的next指针指向的才是单链表的第一个节点，这样的设计可以去除判断头节点为空或存在的判断逻辑。但是如果将这个哨兵节点设计为只包含一个 next域的节点，可以进一步节省一个sizeof(_Tp)的内存空间，要知道这个size可能比较大。
![](images/2021-12-05-00-05-04.png)
