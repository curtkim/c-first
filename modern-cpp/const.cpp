// https://stackoverflow.com/questions/15999123/const-before-parameter-vs-const-after-function-name-c
struct X
{
  void foo() const // <== The implicit "this" pointer is const-qualified!
  {
    //_x = 42; // ERROR! The "this" pointer is implicitly const
    _y = 42; // OK (_y is mutable)
  }

  void bar(X& obj) const // <== The implicit "this" pointer is const-qualified!
  {
    obj._x = 42; // OK! obj is a reference to non-const
    //_x = 42; // ERROR! The "this" pointer is implicitly const
  }

  void bar(X const& obj) // <== The implicit "this" pointer is NOT const-qualified!
  {
    //obj._x = 42; // ERROR! obj is a reference to const
    obj._y = 42; // OK! obj is a reference to const, but _y is mutable
    _x = 42; // OK! The "this" pointer is implicitly non-const
  }

  int _x;
  mutable int _y;
};

int main() {
  return 0;
}