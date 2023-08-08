#include <vector>
#include <iostream>
class Vec {
public:
    static Vec& Instance();
    int Add(int a);
    Vec() {
        std::cout << "i am vec";
    }
private:
    std::vector<int> vec_;
};