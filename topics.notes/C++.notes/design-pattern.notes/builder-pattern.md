```c++
#include <iostream>
#include <string>
#include <cstdarg>

struct Person;

struct LiveBuilder;
struct WorkBuilder;

struct BuilderBase {
    Person& person_;
    BuilderBase(Person& p)
        : person_(p)
    { }

    operator Person();

    LiveBuilder live() const;
    WorkBuilder work() const;
};

struct LiveBuilder : BuilderBase {
    LiveBuilder(Person& p)
        : BuilderBase(p)
    { }

    typedef LiveBuilder self;

    self& name(const std::string& x);
    self& age(int x);
    self& at(const std::string& x);
};

struct WorkBuilder : BuilderBase {
    WorkBuilder(Person& p)
        : BuilderBase(p)
    { }

    typedef WorkBuilder self;

    self& at(const std::string& n);
    self& as_a(const std::string& x);
};

LiveBuilder BuilderBase::live() const {
    return LiveBuilder(person_);
}

WorkBuilder BuilderBase::work() const
{
    return WorkBuilder(person_);
}

struct PersonBuilder;
struct Person {
    std::string name_;
    int age_;
    std::string liveAddress_;
    std::string workAddress_;
    std::string title_;

    static PersonBuilder create();

    friend std::ostream& operator<<(std::ostream& outstr, const Person& p) {
        return outstr << p.name_ << ", "
            << p.age_ << ", "
            << p.liveAddress_ << ", "
            << p.workAddress_ << ", "
            << p.title_;
    }
};

struct PersonBuilder : BuilderBase {
    Person person_;
    PersonBuilder() 
        : BuilderBase(person_)
    { }
};

PersonBuilder Person::create()
{
    return PersonBuilder();
}

BuilderBase::operator Person() {
    return std::move(person_);
}

LiveBuilder::self& LiveBuilder::name(const std::string& x)
{
    person_.name_ = x;
    return *this;
}

LiveBuilder::self& LiveBuilder::age(int x)
{
    person_.age_ = x;
    return *this;
}

LiveBuilder::self& LiveBuilder::at(const std::string& x)
{
    person_.liveAddress_ = x;
    return *this;
}

WorkBuilder::self& WorkBuilder::at(const std::string& x)
{
    person_.workAddress_ = x;
    return *this;
}

WorkBuilder::self& WorkBuilder::as_a(const std::string& x)
{
    person_.title_ = x;
    return *this;
}

int main()
{
    Person p = Person::create()
        .live().name("Leo").age(40)
                .at("live_at_xxx")
        .work().at("work_at_yyy")
                .as_a("engineer");
    std::cout << p << "\n";       
}
```