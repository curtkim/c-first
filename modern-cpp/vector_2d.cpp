#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int main() {
  // Initializing a 2D Vector i.e. vector of vectors
  vector<vector<int>> matrix = {  {1, 2, 3, 4, 5 },
                                  {6, 7, 8, 9, 10 },
                                  {5, 6, 8, 1, 12 },
                                  {1, 7, 2, 4, 18 },
  };

  // Print 2D vector / matrix
  for_each(matrix.begin(), matrix.end(),
           [](const auto & row ) {
             for_each(row.begin(), row.end(),
                      [](const auto & elem){
                        cout<<elem<<", ";
                      });
             cout<<endl;
           });
  cout<<endl;
  return 0;

}