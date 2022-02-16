= Protocol Specification =

The Redis protocol is a compromise between being easy to parse by a computer and being easy to parse by an human. Before reading this section you are strongly encouraged to read the "REDIS TUTORIAL" section of this README in order to get a first feeling of the protocol playing with it by TELNET.
Networking layer
A client connects to a Redis server creating a TCP connection to the port 6379. Every redis command or data transmitted by the client and the server is terminated by "\r\n" (CRLF).
Simple INLINE commands
The simplest commands are the inline commands. This is an example of a server/client chat (the server chat starts with S:, the client chat with C:)

C: PING
S: +PONG
An inline command is a CRLF-terminated string sent to the client. The server can reply to commands in different ways:
With an error message (the first byte of the reply will be "-")
With a single line reply (the first byte of the reply will be "+)
With bulk data (the first byte of the reply will be "$")
With multi-bulk data, a list of values (the first byte of the reply will be "*")
With an integer number (the first byte of the reply will be ":")
The following is another example of an INLINE command returning an integer:

C: EXISTS somekey
S: :0
Since 'somekey' does not exist the server returned ':0'.

Note that the EXISTS command takes one argument. Arguments are separated simply by spaces.
Bulk commands
A bulk command is exactly like an inline command, but the last argument of the command must be a stream of bytes in order to send data to the server. the "SET" command is a bulk command, see the following example:

C: SET mykey 6
C: foobar
S: +OK
The last argument of the commnad is '6'. This specify the number of DATA bytes that will follow (note that even this bytes are terminated by two additional bytes of CRLF).

All the bulk commands are in this exact form: instead of the last argument the number of bytes that will follow is specified, followed by the bytes, and CRLF. In order to be more clear for the programmer this is the string sent by the client in the above sample:

"SET mykey 6\r\nfoobar\r\n"
Bulk replies
The server may reply to an inline or bulk command with a bulk reply. See the following example:

C: GET mykey
S: $6
S: foobar
A bulk reply is very similar to the last argument of a bulk command. The server sends as the first line a "$" byte followed by the number of bytes of the actual reply followed by CRLF, then the bytes are sent followed by additional two bytes for the final CRLF. The exact sequence sent by the server is:

"$6\r\nfoobar\r\n"
If the requested value does not exist the bulk reply will use the special value -1 as data length, example:

C: GET nonexistingkey
S: $-1
The client library API should not return an empty string, but a nil object, when the requested object does not exist. For example a Ruby library should return 'nil' while a C library should return NULL, and so forth.
Multi-Bulk replies
Commands similar to LRANGE needs to return multiple values (every element of the list is a value, and LRANGE needs to return more than a single element). This is accomplished using multiple bulk writes, prefixed by an initial line indicating how many bulk writes will follow. The first byte of a multi bulk reply is always *. Example:

C: LRANGE mylist 0 3
S: *4
S: $3
S: foo
S: $3
S: bar
S: $5
S: Hello
S: $5
S: World
The first line the server sent is "4\r\n" in order to specify that four bulk write will follow. Then every bulk write is transmitted.

If the specified key does not exist instead of the number of elements in the list, the special value -1 is sent as count. Example:

C: LRANGE nokey 0 1
S: *-1
A client library API SHOULD return a nil object and not an empty list when this happens. This makes possible to distinguish between empty list and non existing ones.
Nil elements in Multi-Bulk replies
Single elements of a multi bulk reply may have -1 length, in order to signal that this elements are missing and not empty strings. This can happen with the SORT command when used with the GET pattern option when the specified key is missing. Example of a multi bulk reply containing an empty element:

S: *3
S: $3
S: foo
S: $-1
S: $3
S: bar
The second element is nul. The client library should return something like this:

["foo",nil,"bar"]
Single line reply
As already seen a single line reply is in the form of a single line string starting with "+" terminated by "\r\n". For example:

+OK
The client library should return everything after the "+", that is, the string "OK" in the example.

The following commands reply with a status code reply: PING, SET, SELECT, SAVE, BGSAVE, SHUTDOWN, RENAME, LPUSH, RPUSH, LSET, LTRIM
Integer reply
This type of reply is just a CRLF terminated string representing an integer, prefixed by a ":" byte. For example ":0\r\n", or ":1000\r\n" are integer replies.

With commands like INCR or LASTSAVE using the integer reply to actually return a value there is no special meaning for the returned integer. It is just an incremental number for INCR, a UNIX time for LASTSAVE and so on.

Some commands like EXISTS will return 1 for true and 0 for false.

Other commands like SADD, SREM and SETNX will return 1 if the operation was actually done, 0 otherwise.

The following commands will reply with an integer reply: SETNX, DEL, EXISTS, INCR, INCRBY, DECR, DECRBY, DBSIZE, LASTSAVE, RENAMENX, MOVE, LLEN, SADD, SREM, SISMEMBER, SCARD
Multi bulk commands
As you can see with the protocol described so far there is no way to send multiple binary-safe arguments to a command. With bulk commands the last argument is binary safe, but there are commands where multiple binary-safe commands are needed, like the MSET command that is able to SET multiple keys in a single operation.

In order to address this problem Redis 1.1 introduced a new way of seding commands to a Redis server, that uses exactly the same protocol of the multi bulk replies. For instance the following is a SET command using the normal bulk protocol:

SET mykey 8
myvalue
While the following uses the multi bulk command protocol:

*3
$3
SET
$5
mykey
$8
myvalue
Commands sent in this format are longer, so currently they are used only in order to transmit commands containing multiple binary-safe arguments, but actually this protocol can be used to send every kind of command, without to know if it's an inline, bulk or multi-bulk command.

It is possible that in the future Redis will support only this format.

A good client library may implement unknown commands using this command format in order to support new commands out of the box without modifications.
Multiple commands and pipelining
A client can use the same connection in order to issue multiple commands. Pipelining is supported so multiple commands can be sent with a single write operation by the client, it is not needed to read the server reply in order to issue the next command. All the replies can be read at the end.

Usually Redis server and client will have a very fast link so this is not very important to support this feature in a client implementation, still if an application needs to issue a very large number of commands in short time to use pipelining can be much faster.
