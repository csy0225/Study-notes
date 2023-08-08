#include "save.h"

Vec& Vec::Instance() {
    static Vec vec;
    return vec;
}

int Vec::Add(int i) {
    vec_.push_back(i);
    return i;
}