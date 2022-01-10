# chapter1: Introduction

let’s ask ourselves
a question: why do we violently reject some techniques, even before
studying them?

let’s be frank, sometimes concepts are just plain nonsense
or totally irrelevant to the task at hand.

The purpose of this report is to demonstrate that understanding
C++ metaprogramming will make you a better C++ programmer, as
well as a better software engineer in general.

## A Misunderstood Technique
As you progress along the path of software engineering, the techni‐
ques you learn are more and more advanced. You could opt to rely
solely on simple techniques and solve complex problems via a com‐
position of these techniques, but you will be missing an opportunity
to be more concise, more productive, and sometimes more efficient.

What makes a good software engineer is not only the size of his
toolbox, but, most importantly, his ability to choose the right tool at
the right time.

The concepts behind C++ metaprogramming, on the other hand,
are extremely coherent and logical: **it’s functional programming!**

## What Is Metaprogramming?
By definition, metaprogramming is the design of programs whose
input and output are programs themselves.
Put another way, it’s writing code whose job is to write code itself. It can be seen as the
ultimate level of abstraction, as code fragments are actually seen as
data and handled as such.

## Enter C++ Templates
Templates were originally a very simple feature, allowing code to be parameterized by teyps and integral constants in such a way that more generic code can emere from a collection of existing variants of a given piece of code.

C++ template metaprogramming is a technique based on the use (and abuse) of C++ template properties to perform arbitrary computations at compile time

A classic roster of applications of C++ template metaprogramming includes the
following:
* Complex constant computations
* Programmatic type constructions
* Code fragment generation and replication

Boost.MPL and Boost.Fusion
The main goal of those components was to provide compile-time constructs with an STL look and free. Boost.MPL is designed around compile-time containers, algorithms and iterators. In the same way, Boost.Fusion provides algorithms to work both at compile-time and at runtime on tuple-like structures.
