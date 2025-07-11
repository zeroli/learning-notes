#include <iostream>
#include <type_traits>
#include <string>
#include <vector>
#include <memory>

/// 这个简单的示例代码演示了type erasure设计模式
/// 用来解决依赖性的问题的一个应用
/// square/circle虽然是Shape，但是不再通过继承的方式
/// 来提供一些常用的施行于它们之上的操作函数
/// 而是抽象出一个value类:Shape，来表示所有不同形状
/// 通过模板构造函数使得Shape类可以接收不同具体形状的构造。
struct Square {
    Square(int side) : d_side(side) {}

    int side() const { return d_side; }
private:
    int d_side;
};

struct Circle {
    Circle(int radius) : d_radius(radius) {}

    int radius() const { return d_radius; }
private:
    int d_radius;
};

template <typename T>
struct ShapeTraits {
    static void Draw(const T& t) {
        /// 这里可以通过ADL查找到之后声明的
        /// T类型对应的重载函数
        /// 当然这个函数的实现放到Shape后实现之后也完全没有问题
        draw(t);
    }
};

struct Shape {
    struct Concept {
        virtual ~Concept() = default;

        virtual void draw() const = 0;
    };

    template <typename T>
    struct Model : Concept {
        Model(const T& t)
            : t_(t)
        {}
        virtual ~Model() = default;

        virtual void draw() const override {
            /// 这里不能直接调用`draw(t_);`
            /// 因为编译器看不到之后声明的
            /// `draw(const Cirle&)`和`draw(const Square&)`
            /// 但是通过traits类这个中间层提供的Draw函数声明
            /// 就可以让编译通过
            ShapeTraits<T>::Draw(t_);
        }
        T t_;
    };
    template <typename T>
    Shape(const T& t)
        : d_impl(new Model<T>(t))
    {}

    Shape(const Shape&) = delete;
    Shape& operator =(const Shape&) = delete;
    Shape(Shape&&) = default;
    Shape& operator =(Shape&&) = default;

    // 这里可以用friend函数，也可以用成员函数
    friend void draw(const Shape& shape) {
        shape.d_impl->draw();
    }

private:
    std::unique_ptr<Concept> d_impl;
};

void draw(const Square& square) {
    std::cout << "draw square(" << square.side() << ")\n";
}
void draw(const Circle& circle) {
    std::cout << "draw circle(" << circle.radius() << ")\n";
}

int main()
{
    std::vector<Shape> shapes;
    shapes.emplace_back(Shape(Square(1)));
    shapes.emplace_back(Shape(Circle(2)));
    for (auto&& shape : shapes) {
        draw(shape);
    }
}
