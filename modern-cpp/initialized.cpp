struct A {
    int i;
    bool b;
};

A a1{1, true};            // aggregate initializer
A a2{.i = 10, .b = true}; // designated initializer


