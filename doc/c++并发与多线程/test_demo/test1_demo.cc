#include <iostream>
using namespace std;
#include <thread>
#include <vector>
#include <list>
#include <mutex>

class A {
public:
A() {
    std::cout << "I am construct func;" << std::endl;
}

A (const A& a) {
    std::cout << "I am copy construct func;" << std::endl;
}

~A() {
    std::cout << "I am destruct func;" << std::endl;   
}
};

A func() {
    std::cout << "func is come in;" << std::endl;
    A a;
    return a;
}

std::unique_ptr<int> func1() {
    std::cout << "func is come in;" << std::endl;
    std::unique_ptr<int> a(new int(5));
    return std::move(a);
}
int main() {
    // A myobja = func();
    auto my_ojb = func1();
    std::cout << *my_ojb;
    return 0;
}