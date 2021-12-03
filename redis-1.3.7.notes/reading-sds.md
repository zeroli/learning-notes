**date: 11/10/2021**

# 这个主题是讨论redis里面的sds (simple dynamic string)数据结构类型

**file: sds.h/c**

## Redis String Type
>
> Strings are the most basic Redis kind of values. Redis Strings are binary safe, this means a Redis string can contain any kind of data, for instance a JPEG image or a serialized Ruby object, and so forth.

> A String value can be at max 1 Gigabyte in length.

> Strings are treated as integer values by the INCR commands family, in this respect the value of an intger is limited to a singed 64 bit value.

> Note that the single elements contained in Redis Lists, Sets and Sorted Sets, are Redis Strings.
> Implementation details
> Strings are implemented using a dynamic strings library called sds.c (simple dynamic strings). This library caches the current length of the string, so to obtain the length of a Redis string is an O(1) operation (but currently there is no such STRLEN command. It will likely be added later).

> Redis strings are incapsualted into Redis Objects. Redis Objects use a reference counting memory management system, so a single Redis String can be shared in different places of the dataset. This means that if you happen to use the same strings many times (especially if you have object sharing turned on in the configuration file) Redis will try to use the same string object instead to allocate one new every time.

> Starting from version 1.1 Redis is also able to encode in a special way strings that are actually just numbers. Instead to save the string as an array of characters Redis will save the integer value in order to use less memory. With many datasets this can reduce the memory usage of about 30% compared to Redis 1.0.

* sds结构体定义如下：
```c
typedef char *sds;

struct sdshdr {
    long len;
    long free;
    char buf[];
};
```
所以`sds`操作类型，其实就是一个char*，指向`sdshdr`结构体里面的`buf`，但是因为在它前面有`len`/`free`用来记录`buf`的使用情况，故而可以在O(1)复杂度内计算出字符串的长度。

接下来我们看下它的一些API函数:
`sds sdsnew(const char* init, size_t initlen);`
```c
sds sdsnewlen(const void *init, size_t initlen) {
    struct sdshdr *sh;

    sh = zmalloc(sizeof(struct sdshdr)+initlen+1);
#ifdef SDS_ABORT_ON_OOM
    if (sh == NULL) sdsOomAbort();
#else
    if (sh == NULL) return NULL;
#endif
    sh->len = initlen;
    sh->free = 0;
    if (initlen) {
        if (init) memcpy(sh->buf, init, initlen);
        else memset(sh->buf,0,initlen);
    }
    sh->buf[initlen] = '\0';
    return (char*)sh->buf;
}
```
以一个C字符串来初始化这个`sds`
- 分配空间 `sizeof(struct sdshdr)+initlen+1` (header size + 字符串本身大小 + '\0')
- 初始化header结构体的变量
- 拷贝C字符串到这个新的内存空间
- 总是以'\0'结尾，这样`sds`便可以于C的库函数无缝衔接
- 返回`sh->buf`作为`sds`

* sdslen
```c
size_t sdslen(const sds s) {
    struct sdshdr *sh = (void*) (s-(sizeof(struct sdshdr)));
    return sh->len;
}
```
将`sds`变量直接shift到header头，然后获取它的len


当sds空间不够时，需要resize，它的策略就是简单double一下：
```c
static sds sdsMakeRoomFor(sds s, size_t addlen) {
    struct sdshdr *sh, *newsh;
    size_t free = sdsavail(s);
    size_t len, newlen;

    if (free >= addlen) return s;  // 还有剩余空间
    len = sdslen(s);
    sh = (void*) (s-(sizeof(struct sdshdr)));
    newlen = (len+addlen)*2;  // 简单的double一下大小
    newsh = zrealloc(sh, sizeof(struct sdshdr)+newlen+1); // 调用realloc来重新分配更大的内存空间
#ifdef SDS_ABORT_ON_OOM
    if (newsh == NULL) sdsOomAbort();
#else
    if (newsh == NULL) return NULL;
#endif

    newsh->free = newlen - len;
    return newsh->buf;
}
```
上面这个函数是静态函数，被很多API调用，譬如`sdscatlen`函数，concatenate另外一个字符串:
```c
sds sdscatlen(sds s, void *t, size_t len) {
    struct sdshdr *sh;
    size_t curlen = sdslen(s);

    s = sdsMakeRoomFor(s,len);  // 能否resize??
    if (s == NULL) return NULL;
    sh = (void*) (s-(sizeof(struct sdshdr)));
    memcpy(s+curlen, t, len);  // 拷贝字符串到尾巴那里
    sh->len = curlen+len;
    sh->free = sh->free-len;
    s[curlen+len] = '\0';  // 总是null-terminate
    return s;
}
```

concatenate还支持concatenate一个要格式化的字符串：
```c
sds sdscatprintf(sds s, const char *fmt, ...) {
    va_list ap;
    char *buf, *t;
    size_t buflen = 16;  // 以16作为初始值

    while(1) {
        buf = zmalloc(buflen);
        buf[buflen-2] = '\0';
        va_start(ap, fmt);
        vsnprintf(buf, buflen, fmt, ap);
        va_end(ap);
        if (buf[buflen-2] != '\0') {  // 如果vsnprintf不成功
            zfree(buf);
            buflen *= 2;  // double its size
            continue;  // 重新尝试
        }
        break;
    }
    t = sdscat(s, buf);
    zfree(buf);
    return t;
}
```
以上这种方法比较有效率，有可能要格式化的字符串比较短，这样的方式就可以一次性成功。
