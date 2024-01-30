#include <iostream>
#include "p3.h"
using namespace std;

int main() {

    PStash<int> obj;

    obj.add(new int(1));
    obj.add(new int(2));
    obj.add(new int(3));
    obj.add(new int(4));

    for (auto o : obj ) {
        cout << *o << "A" << endl;
    }
}