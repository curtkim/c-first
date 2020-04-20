#include <stdio.h>

#define N (1024*1024)
#define M (1000000)

int main()
{
  float data[N]; int count = 0;
  for(int i = 0; i < N; i++)
  {
    data[i] = 1.0f * i / N;
    for(int j = 0; j < M; j++)
    {
      data[i] = data[i] * data[i] - 0.25f;
    }
  }
  int sel;
  printf("Enter an index: ");
  scanf("%d", &sel);
  printf("data[%d] = %f\n", sel, data[sel]);
}
