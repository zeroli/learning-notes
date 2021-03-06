## = Append Only File HOWTO =

General Information
Append only file is an alternative durability option for Redis. What this mean? Let's start with some fact:

For default Redis saves snapshots of the dataset on disk, in a binary file called dump.rdb (by default at least). For instance you can configure Redis to save the dataset every 60 seconds if there are at least 100 changes in the dataset, or every 1000 seconds if there is at least a single change in the dataset. This is known as "Snapshotting".
Snapshotting is not very durable. If your computer running Redis stops, your power line fails, or you write killall -9 redis-server for a mistake, the latest data written on Redis will get lost. There are applications where this is not a big deal. There are applications where this is not acceptable and Redis was not an option for this applications.
What is the solution? To use append only file as alternative to snapshotting. How it works?

It is an 1.1 only feature.
You have to turn it on editing the configuration file. Just make sure you have "appendonly yes" somewhere.
Append only files work this way: every time Redis receive a command that changes the dataset (for instance a SET or LPUSH command) it appends this command in the append only file. When you restart Redis it will first re-play the append only file to rebuild the state.
Log rewriting
As you can guess... the append log file gets bigger and bigger, every time there is a new operation changing the dataset. Even if you set always the same key "mykey" to the values of "1", "2", "3", ... up to 10000000000 in the end you'll have just a single key in the dataset, just a few bytes! but how big will be the append log file? Very very big.

**So Redis supports an interesting feature: it is able to rebuild the append log file, in background, without to stop processing client commands. The key is the command BGREWRITEAOF. This command basically is able to use the dataset in memory in order to rewrite the shortest sequence of commands able to rebuild the exact dataset that is currently in memory.**

So from time to time when the log gets too big, try this command. It's safe as if it fails you will not lost your old log (but you can make a backup copy given that currently 1.1 is still in beta!).
Wait... but how does this work?
Basically it uses the same fork() copy-on-write trick that snapshotting already uses. This is how the algorithm works:

  - Redis forks, so now we have a child and a parent.
  - The child starts writing the new append log file in a temporary file.
  - The parent accumulates all the new changes in an in-memory buffer.
  - When the child finished to rewrite the file, the parent gets a signal, and append the in-memory buffer at the end of the file generated by the child.
  - Profit! Now Redis atomically renames the old file into the new one, and starts appending new data into the new file.
How durable is the append only file?
Check redis.conf, you can configure how many times Redis will fsync() data on disk. There are three options:

- Fsync() every time a new command is appended to the append log file. Very very slow, very safe.
- Fsync() one time every second. Fast enough, and you can lose 1 second of data if there is a disaster.
- Never fsync(), just put your data in the hands of the Operating System. The faster and unsafer method.
Warning: by default Redis will fsync() after every command! This is because the Redis authors want to ship a default configuration that is the safest pick. But the best compromise for most datasets is to fsync() one time every second.
