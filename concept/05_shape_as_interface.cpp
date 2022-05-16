// from https://www.cppfiddler.com/2019/06/09/concept-based-interfaces/
// Shape
// ---------
// Rectangle
// Square
// Circle

#include <concepts>
#include <cstdio>

template <typename T>
concept Shape = requires(const T& t)
{
  { t.area() } ->std::convertible_to<float>;
};

struct Rectangle
{
  Rectangle()
  {
    static_assert(Shape<decltype(*this)>);
  }
  float area() const {
    return 2.0;
  };
};


template <typename T>
struct Square
{
  Square(T edge) : edge(edge)
  {
    static_assert(Shape<decltype(*this)>);
  }
  float area() const {
    return edge*edge;
  }
  T edge;
};

template <typename T>
struct ShapeBase
{
  ShapeBase() { static_assert(Shape<T>); }
};


template <typename T>
struct Circle : ShapeBase<Circle<T>>
{
  float area() const{
    return 3.14 * radius * radius;
  };
  T radius;
};


template <Shape S>
void print_area(S s){
  printf("%f\n", s.area());
}

int main()
{
  Rectangle r;
  print_area(r);

  Square<int> s{1};
  print_area(s);

  Circle<float> c;
  c.radius = 1.0;
  print_area(c);

  return 0;
}