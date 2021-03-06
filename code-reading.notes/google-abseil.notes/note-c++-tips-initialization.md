C++11 provided a new syntax referred to as “uniform initialization syntax” that was supposed to unify all of the various styles of initialization, avoid the Most Vexing Parse, and avoid narrowing conversions. This new mechanism means we now have yet another syntax for initialization, with its own tradeoffs.

C++11 Brace Initialization
=======
Some uniform initialization syntax proponents would suggest that we use {}s and direct initialization (no use of the ‘=’, although in most cases both forms call the same constructor) for initialization of all types:

```c++
int x{2};
std::string foo{"Hello World"};
std::vector<int> v{1, 2, 3};
```
vs. (for instance):
```c++
int x = 2;
std::string foo = "Hello World";
std::vector<int> v = {1, 2, 3};
```
This approach has two shortcomings. First, “uniform” is a stretch: there are cases where ambiguity still exists (for the casual reader, not the compiler) in what is being called and how.
```c++
std::vector<std::string> strings{2}; // A vector of two empty strings.
std::vector<int> ints{2};            // A vector containing only the integer 2.
```
Second: this syntax is not exactly intuitive: no other common language uses something like it. The language can certainly introduce new and surprising syntax, and there are technical reasons why it’s necessary in some cases – especially in generic code. The important question is: how much should we change our habits and language understanding to take advantage of that change? Are the benefits worth the cost in changing our habits or our existing code? For uniform initialization syntax, we don’t believe in general that the benefits outweigh the drawbacks.

* Best Practices for Initialization
Instead, we recommend the following guidelines for “How do I initialize a variable?”, both to follow in your own code and to cite in your code reviews:

Use assignment syntax when initializing directly with the intended literal value (for example: int, float, or std::string values), with smart pointers such as std::shared_ptr, std::unique_ptr, with containers (std::vector, std::map, etc), when performing struct initialization, or doing copy construction.
```c++
int x = 2;
std::string foo = "Hello World";
std::vector<int> v = {1, 2, 3};
std::unique_ptr<Matrix> matrix = NewMatrix(rows, cols);
MyStruct x = {true, 5.0};
MyProto copied_proto = original_proto;
```
instead of:
```c++
// Bad code
int x{2};
std::string foo{"Hello World"};
std::vector<int> v{1, 2, 3};
std::unique_ptr<Matrix> matrix{NewMatrix(rows, cols)};
MyStruct x{true, 5.0};
MyProto copied_proto{original_proto};
```
Use the traditional constructor syntax (with parentheses) when the initialization is performing some active logic, rather than simply composing values together.
```c++
Frobber frobber(size, &bazzer_to_duplicate);
std::vector<double> fifty_pies(50, 3.14);
```
vs.
```c++
// Bad code

// Could invoke an intializer list constructor, or a two-argument constructor.
Frobber frobber{size, &bazzer_to_duplicate};

// Makes a vector of two doubles.
std::vector<double> fifty_pies{50, 3.14};
```
Use {} initialization without the = only if the above options don’t compile:
```c++
class Foo {
 public:
  Foo(int a, int b, int c) : array_{a, b, c} {}

 private:
  int array_[5];
  // Requires {}s because the constructor is marked explicit
  // and the type is non-copyable.
  EventManager em{EventManager::Options()};
};
```
Never mix {}s and auto.
For example, don’t do this:
```c++
// Bad code
auto x{1};
auto y = {2}; // This is a std::initializer_list<int>!
```
(For the language lawyers: prefer copy-initialization over direct-initialization when available, and use parentheses over curly braces when resorting to direct-initialization.)

Perhaps the best overall description of the issue is Herb Sutter’s GotW post. Although he shows examples that include direct initialization of int with braces, his final advice is roughly compatible with what we present here with one caveat: where Herb says “where you prefer to see only the = sign”, we unambiguously prefer to see exactly that. In conjunction with more consistent use of explicit on multi-parameter constructors (see Tip #142), this provides a balance between readability, explicitness, and correctness.

Conclusion
======
The tradeoffs for uniform initialization syntax are not generally worth it: our compilers already warn against the Most Vexing Parse (you can use brace initialization or add parens to resolve the issue), and the safety from narrowing conversions isn’t worth the readability hit for brace-initialization (we’ll need a different solution for narrowing conversions, eventually). The Style Arbiters don’t think this issue is critical enough to make a formal rule on, especially because there are cases (notably in generic code) where brace initialization may be justified.
