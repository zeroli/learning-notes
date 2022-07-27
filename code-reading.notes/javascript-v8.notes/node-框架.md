# V8执行javascript框架

V8执行javascript代码的框架如下：
```c++
int RunMain(int argc, char* argv[]) {
  v8::V8::SetFlagsFromCommandLine(&argc, argv, true);
  v8::HandleScope handle_scope;
  // Create a template for the global object.
  v8::Handle<v8::ObjectTemplate> global = v8::ObjectTemplate::New();
  // Bind the global 'print' function to the C++ Print callback.
  global->Set(v8::String::New("print"), v8::FunctionTemplate::New(Print));
  // Bind the global 'read' function to the C++ Read callback.
  global->Set(v8::String::New("read"), v8::FunctionTemplate::New(Read));
  // Bind the global 'load' function to the C++ Load callback.
  global->Set(v8::String::New("load"), v8::FunctionTemplate::New(Load));
  // Bind the 'quit' function
  global->Set(v8::String::New("quit"), v8::FunctionTemplate::New(Quit));
  // Bind the 'version' function
  global->Set(v8::String::New("version"), v8::FunctionTemplate::New(Version));
  // Create a new execution environment containing the built-in
  // functions
  v8::Handle<v8::Context> context = v8::Context::New(NULL, global);
  // Enter the newly created execution environment.
  v8::Context::Scope context_scope(context);
  bool run_shell = (argc == 1);
  for (int i = 1; i < argc; i++) {
    const char* str = argv[i];
    if (strcmp(str, "--shell") == 0) {
      run_shell = true;
    } else if (strcmp(str, "-f") == 0) {
      // Ignore any -f flags for compatibility with the other stand-
      // alone JavaScript engines.
      continue;
    } else if (strncmp(str, "--", 2) == 0) {
      printf("Warning: unknown flag %s.\nTry --help for options\n", str);
    } else if (strcmp(str, "-e") == 0 && i + 1 < argc) {
      // Execute argument given to -e option directly
      v8::HandleScope handle_scope;
      v8::Handle<v8::String> file_name = v8::String::New("unnamed");
      v8::Handle<v8::String> source = v8::String::New(argv[i + 1]);
      if (!ExecuteString(source, file_name, false, true))
        return 1;
      i++;
    } else {
      // Use all other arguments as names of files to load and run.
      v8::HandleScope handle_scope;
      v8::Handle<v8::String> file_name = v8::String::New(str);
      v8::Handle<v8::String> source = ReadFile(str);
      if (source.IsEmpty()) {
        printf("Error reading '%s'\n", str);
        return 1;
      }
      if (!ExecuteString(source, file_name, false, true))
        return 1;
    }
  }
  if (run_shell) RunShell(context);
  return 0;
}
```

这种方式非常类似于Lua
* 将一些内置C++函数注入到全局环境中`global`;
```c++
v8::Handle<v8::Value> Print(const v8::Arguments& args);
v8::Handle<v8::Value> Read(const v8::Arguments& args);
v8::Handle<v8::Value> Load(const v8::Arguments& args);
v8::Handle<v8::Value> Quit(const v8::Arguments& args);
v8::Handle<v8::Value> Version(const v8::Arguments& args);
```
它们得符合函数签名：参数是`const v8::Argument&`，返回值类型是`v8::Handle<v8::Value>`
* 几个重要的类：
    * `ObjectTemplate`
    * `FunctionTemplate`
    * `String`
    * `Object`
    * `Context`
    * `Context::Scope`
    * `Handle`
    * `HandleScope`


如何执行Javascript代码？
===
```c++
bool ExecuteString(v8::Handle<v8::String> source,
                   v8::Handle<v8::Value> name,
                   bool print_result,
                   bool report_exceptions) {
  v8::HandleScope handle_scope;
  v8::TryCatch try_catch;
  v8::Handle<v8::Script> script = v8::Script::Compile(source, name);
  if (script.IsEmpty()) {
    // Print errors that happened during compilation.
    if (report_exceptions)
      ReportException(&try_catch);
    return false;
  } else {
    v8::Handle<v8::Value> result = script->Run();
    if (result.IsEmpty()) {
      // Print errors that happened during execution.
      if (report_exceptions)
        ReportException(&try_catch);
      return false;
    } else {
      if (print_result && !result->IsUndefined()) {
        // If all went well and the result wasn't undefined then print
        // the returned value.
        v8::String::Utf8Value str(result);
        const char* cstr = ToCString(str);
        printf("%s\n", cstr);
      }
      return true;
    }
  }
}
```
* 编译：`v8::Handle<v8::Script> script = v8::Script::Compile(source, name);`
* 执行：`v8::Handle<v8::Value> result = script->Run();`
* 报告编译或执行期间出现的异常，如果函数调用时要求报告异常的话。
