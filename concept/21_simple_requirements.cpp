template<typename T>
concept Addable = requires(T a, T b){
  a + b;
};

auto add(Addable auto a, Addable auto b){
  return a+b;
}

int main(){
  add(1, 2);
  add(1.1, 2.2);
  add("a", "b");
}