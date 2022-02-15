**date: 11/10/2021**

# redis-1.3.7代码阅读

## Introduction
Redis is a database. To be specific, Redis is a database implementing a dictionary, where every key is associated with a value. For example I can set the key "surname_1992" to the string "Smith". What makes Redis different from many other key-value stores, is that every single value has a type. The following types are supported:


-   [Strings](Strings.html)
-   [Lists](Lists.html)
-   [Sets](Sets.html)
-   [Sorted Set](SortedSets.html)  (since version 1.1)

The type of a value determines what operations (called commands) are available for the value itself. For example you can append elements to a list stored at the key "mylist" using the LPUSH or RPUSH command in O(1). Later you'll be able to get a range of elements with LRANGE or trim the list with LTRIM. Sets are very flexible too, it is possible to add and remove elements from Sets (unsorted collections of strings), and then ask for server-side intersection, union, difference of Sets. Each command is performed through server-side atomic operations. Please refer to the [Command Reference](CommandReference.html) to see the full list of operations associated to these data types.

In other words, you can look at Redis as a data structures server. A Redis user is virtually provided with an interface to [Abstract Data Types](http://en.wikipedia.org/wiki/Abstract_data_type), saving her from the responsibility to implement concrete data structures and algorithms. Indeed both algorithms and data structures in Redis are properly choosed in order to obtain the best performance.

## file: redis-cli.c
这是redis的客户端代码，main函数:
```c++
int  main(int  argc, char  **argv) {
	int  firstarg;
	char  **argvcopy;
	struct  redisCommand  *rc;  // 定义一个redisCommand
    // 一些配置
	config.hostip  =  "127.0.0.1";
	config.hostport  =  6379;
	config.repeat  =  1;
	config.dbnum  =  0;
	config.interactive  =  0;
	config.auth  =  NULL;

	firstarg  =  parseOptions(argc,argv);  // 解析用户输入的参数选项
	argc  -=  firstarg;
	argv  +=  firstarg;
    // 没有参数或者交互模式
	if (argc  ==  0  ||  config.interactive  ==  1) repl();

	argvcopy  =  convertToSds(argc, argv);

	/* Read the last argument from stdandard input if needed */
	if ((rc  =  lookupCommand(argv[0])) !=  NULL) {
	  if (rc->arity  >  0  &&  argc  ==  rc->arity-1) {
	    sds  lastarg  =  readArgFromStdin();
	    argvcopy[argc] =  lastarg;
	    argc++;
	  }
	}

	return  cliSendCommand(argc, argvcopy);
}
```

* config结构是这样定义的：
```c++
static struct config {
    char *hostip;  // 主机地址和端口
    int hostport;
    long repeat;
    int dbnum;  // ????? (TODO）
    int interactive;  // 是否开启交互模式
    char *auth;
} config;
```
* redisCommand结构是这样定义的：
```c++
struct redisCommand {
    char *name;  // 命令的名字
    int arity;  // ????? (TODO)
    int flags;
};
```

* lookupCommand: 根据名字从lookup table `cmdTable`里面查找command
```c++
static struct redisCommand *lookupCommand(char *name) {
    int j = 0;
    while(cmdTable[j].name != NULL) {
        // 进行字符串的比较： strcasecmp，大小写敏感
        if (!strcasecmp(name,cmdTable[j].name)) return &cmdTable[j];
        j++;
    }
    return NULL;
}
```
* 命令lookup table：`cmdTable`
静态初始化一个redisCommand结构体数组： 名字，arity，和flags
```c++
static struct redisCommand cmdTable[] = {
    {"auth",2,REDIS_CMD_INLINE},
    {"get",2,REDIS_CMD_INLINE},
    {"set",3,REDIS_CMD_BULK},
    {"setnx",3,REDIS_CMD_BULK},
    {"append",3,REDIS_CMD_BULK},
    {"substr",4,REDIS_CMD_INLINE},
    {"del",-2,REDIS_CMD_INLINE},
    {"exists",2,REDIS_CMD_INLINE},
    ...
};
```
* cliSendCommand，客户端发送命令给服务器
```c++
static int cliSendCommand(int argc, char **argv) {
    struct redisCommand *rc = lookupCommand(argv[0]);
    int fd, j, retval = 0;
    int read_forever = 0;
    sds cmd;

    if (!rc) {
        fprintf(stderr,"Unknown command '%s'\n",argv[0]);
        return 1;
    }

    if ((rc->arity > 0 && argc != rc->arity) ||
        (rc->arity < 0 && argc < -rc->arity)) {
            fprintf(stderr,"Wrong number of arguments for '%s'\n",rc->name);
            return 1;
    }
    if (!strcasecmp(rc->name,"monitor")) read_forever = 1;
    if ((fd = cliConnect()) == -1) return 1;  // 这里客户端会连接服务器！！！

    /* Select db number */
    retval = selectDb(fd);  // 选择db编号
    if (retval) {
        fprintf(stderr,"Error setting DB num\n");
        return 1;
    }

    while(config.repeat--) {
        /* Build the command to send */
        cmd = sdsempty();
        if (rc->flags & REDIS_CMD_MULTIBULK) {
            cmd = sdscatprintf(cmd,"*%d\r\n",argc);
            for (j = 0; j < argc; j++) {
                cmd = sdscatprintf(cmd,"$%lu\r\n",
                    (unsigned long)sdslen(argv[j]));
                cmd = sdscatlen(cmd,argv[j],sdslen(argv[j]));
                cmd = sdscatlen(cmd,"\r\n",2);
            }
        } else {
            for (j = 0; j < argc; j++) {
                if (j != 0) cmd = sdscat(cmd," ");
                if (j == argc-1 && rc->flags & REDIS_CMD_BULK) {
                    cmd = sdscatprintf(cmd,"%lu",
                        (unsigned long)sdslen(argv[j]));
                } else {
                    cmd = sdscatlen(cmd,argv[j],sdslen(argv[j]));
                }
            }
            cmd = sdscat(cmd,"\r\n");
            if (rc->flags & REDIS_CMD_BULK) {
                cmd = sdscatlen(cmd,argv[argc-1],sdslen(argv[argc-1]));
                cmd = sdscatlen(cmd,"\r\n",2);
            }
        }
        anetWrite(fd,cmd,sdslen(cmd));
        sdsfree(cmd);

        while (read_forever) {
            cliReadSingleLineReply(fd,0);
        }

        retval = cliReadReply(fd);
        if (retval) {
            return retval;
        }
    }
    return 0;
}
```

* 选择db: **selectDB**
通过发送一个命令给服务器：
```c++
static int selectDb(int fd) {
    int retval;
    sds cmd;  // redis自己包装的string结构
    char type;

    if (config.dbnum == 0)  // db num从1开始
        return 0;

    cmd = sdsempty();
    cmd = sdscatprintf(cmd,"SELECT %d\r\n",config.dbnum);  // 建立command string
    anetWrite(fd,cmd,sdslen(cmd));  // 发送command到服务器
    anetRead(fd,&type,1);  // 读取服务器的reply
    if (type <= 0 || type != '+') return 1;  // 期待reply第一个字符是"+"
    retval = cliReadSingleLineReply(fd,1);  // 读取单行的字符串结果
    if (retval) {
        return retval;
    }
    return 0;  // selectDb命令执行成功
}
```

* 客户端连接服务器：**cliConnect**
```c++
static int cliConnect(void) {
    char err[ANET_ERR_LEN];
    static int fd = ANET_ERR;

    if (fd == ANET_ERR) {
        fd = anetTcpConnect(err,config.hostip,config.hostport);
        if (fd == ANET_ERR) {
            fprintf(stderr, "Could not connect to Redis at %s:%d: %s", config.hostip, config.hostport, err);
            return -1;
        }
        anetTcpNoDelay(NULL,fd);
    }
    return fd;
}
```

所以整个流程就是
1. 客户端跟服务器建立连接
2. 告诉服务器选择db num
3. 组装命令然后发送
   * 每个命令是以"\r\n"为结束符
   * 对于字符串内容，以它的长度打头，接着"\r\n"，然后它的字符串内容
   * 发送命令用`anetWrite`函数
4. 同步读取服务器的回复

* socket通信的几个API
  - anetTcpConnect
  - anetTcpNoDelay
  - anetWrite
  - anetRead
