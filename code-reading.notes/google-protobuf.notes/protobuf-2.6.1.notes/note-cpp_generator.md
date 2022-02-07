# protobuf-2.6.1源码阅读

## CPP generator的代码阅读
file: src\google\protobuf\compiler\cpp\cpp_generator.h
file: src\google\protobuf\compiler\cpp\cpp_generator.cc

`CppGenerator`继承于`CodeGenerator`，实现接口函数`Generate`:
```c++
  bool Generate(const FileDescriptor* file,
                const string& parameter,
                GeneratorContext* generator_context,
                string* error) const;
```
传入`FileDescriptor`，将结果输出到`GeneratorContext`的内部buffer中，之后一次性flush到磁盘。
这个Generate函数的基本结构如下：
```c++
  string basename = StripProto(file->name());
  basename.append(".pb");

  FileGenerator file_generator(file, file_options);

  // Generate header.
  {
    scoped_ptr<io::ZeroCopyOutputStream> output(
        generator_context->Open(basename + ".h"));
    io::Printer printer(output.get(), '$');
    file_generator.GenerateHeader(&printer);
  }

  // Generate cc file.
  {
    scoped_ptr<io::ZeroCopyOutputStream> output(
        generator_context->Open(basename + ".cc"));
    io::Printer printer(output.get(), '$');
    file_generator.GenerateSource(&printer);
  }
```
借用`FileGenerator`产生头文件和实现文件。
`io::Printer`是中间的一个间接桥梁类。传入file_generator进行格式化输出，它的backend是output stream。
这里`FileDescriptor`并不跟`ZeroCopyOutputStream`talk，而是让`file_generator`跟`Printer`talk。

Printer做的事情比较简单，仅仅是文本变量替换。
//   Printer printer(output, '$');
//   map<string, string> vars;
//   vars["name"] = "Bob";
//   printer.Print(vars, "My name is $name$.");
//
// The above writes "My name is Bob." to the output stream.

如何生成具体的header/cc file，代码在：
src\google\protobuf\compiler\cpp\cpp_file.h
src\google\protobuf\compiler\cpp\cpp_file.cc
