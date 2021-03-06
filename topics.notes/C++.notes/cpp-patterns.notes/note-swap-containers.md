# swap containers

```c++ {.numberLines}
#include <algorithm>
#include <utility>
#include <vector>

int main()
{
  std::vector<int> v1 = { 1, 3, 5, 7 };
  std::vector<int> v2 = { 2, 4, 6, 8 };

  std::swap_ranges(std::begin(v1), std::end(v1), std::begin(v2));

  v1.swap(v2);

  using std::swap;
  swap(v1, v2);
}
```

INTENT
Swap the contents of two containers.

DESCRIPTION
On lines 7–8, we create two std::vectors and initialise them with values.

On line 10, we swap the contents of the two std::vectors using the std::swap_ranges algorithm, which is given the beginning and end iterators of one range and the beginning iterator of a second range. This performs an element-by-element swap of their values (in this case, four swaps). The second range must have at least the number of elements in the first range. After this operation, iterators remain valid and point at the same container as they did previously.

On line 12, we use std::vector’s swap member function to swap the contents all in one shot. Alternatively, lines 14–15 perform the same operation with the generic swap pattern. This approach is much faster than std::swap_ranges, as it simply swaps over the internal storage in constant time. It also does not require the containers to have the same size. However, after the operation, iterators will now point at the other container and past-the-end iterators are invalidated.
