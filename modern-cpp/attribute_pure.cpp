[[gnu::pure]]
int square(int x){
  return x * x;
}

int main(int argc, char *argv[])
{
  auto x = square(argc);
  auto y = square(argc);
  auto z = x+y;

  return 0;
}