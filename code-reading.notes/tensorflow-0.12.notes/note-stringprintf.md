# stringprintf

```cpp
namespace tensorflow {
namespace strings {

#ifdef COMPILER_MSVC
enum { IS_COMPILER_MSVC = 1 };
#else
enum { IS_COMPILER_MSVC = 0 };
#endif

void Appendv(string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer
  static const int kSpaceLength = 1024;
  char space[kSpaceLength];

  // It's possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  // 这里返回的是格式化字符串的最终的长度(不包括'\0')
  // 如果result < kSpaceLength (最多kSpaceLength)，代表所有格式化的字符串都放入了space
  // 否则，需要更大的空间
  int result = vsnprintf(space, kSpaceLength, format, backup_ap);
  va_end(backup_ap);

  if (result < kSpaceLength) {
    if (result >= 0) {
      // Normal case -- everything fit.
      dst->append(space, result);
      return;
    }

    if (IS_COMPILER_MSVC) {
      // Error or MSVC running out of space.  MSVC 8.0 and higher
      // can be asked about space needed with the special idiom below:
      va_copy(backup_ap, ap);
      result = vsnprintf(NULL, 0, format, backup_ap);
      va_end(backup_ap);
    }

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  int length = result + 1;
  char* buf = new char[length];
  // 但是这里为啥不直接resize dst呢？
  // 在大部分情况下，下面的代码都会ok
  // 这样的话，我们就不需要allocate memory 2 次了，一次是pure buf，然后append到dst 字符串中了。

  // Restore the va_list before we use it again
  va_copy(backup_ap, ap);
  result = vsnprintf(buf, length, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) {
    // It fit
    dst->append(buf, result);
  }
  delete[] buf;
}

string Printf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  Appendv(&result, format, ap);
  va_end(ap);
  return result;
}

void Appendf(string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  Appendv(dst, format, ap);
  va_end(ap);
}
```
