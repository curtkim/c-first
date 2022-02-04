#include <iostream>
#include <chrono>

// loop1 accesses data in matrix 'a' in row major order,
// since i is the outer loop variable, and j is the
// inner loop variable.
int loop1(int a[4000][4000]) {
 int s = 0;
 for (int i = 0; i < 4000; ++i) {
   for (int j = 0; j < 4000; ++j) {
     s += a[i][j];
   }
 }
 return s;
}

// loop2 accesses data in matrix 'a' in column major order
// since j is the outer loop variable, and i is the
// inner loop variable.
int loop2(int a[4000][4000]) {
 int s = 0;
 for (int j = 0; j < 4000; ++j) {
   for (int i = 0; i < 4000; ++i) {
     s += a[i][j];
   }
 }
 return s;
}

int main() {
 static int a[4000][4000] = {0};
 for (int i = 0; i < 100; ++i) {
   int x = rand() % 4000;
   int y = rand() % 4000;
   a[x][y] = rand() % 1000;
 }

 auto start = std::chrono::high_resolution_clock::now();
 auto end = start;
 int s = 0;

#if defined RUN_LOOP1
 start = std::chrono::high_resolution_clock::now();

 s = 0;
 for (int i = 0; i < 10; ++i) {
   s += loop1(a);
   s = s % 100;
 }
 end = std::chrono::high_resolution_clock::now();

 std::cout << "s = " << s << std::endl;
 std::cout << "Time for loop1: "
   << std::chrono::duration<double, std::milli>(end - start).count()
   << "ms" << std::endl;
#endif

#if defined RUN_LOOP2
 start = std::chrono::high_resolution_clock::now();
 s = 0;
 for (int i = 0; i < 10; ++i) {
   s += loop2(a);
   s = s % 100;
 }
 end = std::chrono::high_resolution_clock::now();

 std::cout << "s = " << s << std::endl;
 std::cout << "Time for loop2: "
   << std::chrono::duration<double, std::milli>(end - start).count()
   << "ms" << std::endl;
#endif
}
