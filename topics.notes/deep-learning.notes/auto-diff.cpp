#include <type_traits>
#include <iostream>

struct Expression {
    double value_;

    virtual double eval() { return value_; }
    virtual void derive(double seed) = 0;
};

struct Variable : Expression {
    double partial = 0;

    Variable(double value) {
        value_ = value;
    }

    double eval() override {
        return value_;
    }

    void derive(double seed) override {
        partial += seed;
    }
};

struct PlusOp : Expression {
    Expression *left_ = 0, *right_ = 0;

    PlusOp(Expression* left, Expression* right)
        : left_(left), right_(right)
    { }

    double eval() override {
        value_ = left_->eval() + right_->eval();
        return value_;
    }

    void derive(double seed) override {
        // z = x + y
        // dz/dx = 1, dz/dy = 1
        left_->derive(seed * 1);
        right_->derive(seed * 1);
    }
};

struct MultiplyOp : Expression {
    Expression *left_ = 0, *right_ = 0;

    MultiplyOp(Expression* left, Expression* right)
        : left_(left), right_(right)
    { }

    double eval() override {
        value_ = left_->eval() * right_->eval();
        return value_;
    }

    void derive(double seed) override {
        // z = x * y
        // dz/dx = y, dz/dy = x
        left_->derive(seed * right_->value_);
        right_->derive(seed * left_->value_);
    }
};

int main()
{
    Variable x(2), y(3);
    PlusOp p1(&x, &y);
    MultiplyOp m1(&x, &p1);
    MultiplyOp m2(&y, &y);
    PlusOp z(&m1, &m2);
    std::cout << "z = " << z.eval() << "\n";
    z.derive(1);
    std::cout << "dz / dx = " << x.partial << "\n";
    std::cout << "dz / dy = " << y.partial << "\n";
}

