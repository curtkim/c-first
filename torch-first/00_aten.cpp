#include <ATen/ATen.h>
#include <iostream>

using namespace std;
using namespace at;

void create() {
  // eye
  auto eye = at::eye(2, kInt);

  assert(tensor({
    1,0,
    0,1
  }).reshape({2,2}).equal(eye));
  assert(kInt == eye.options().dtype());
  assert(2 == eye.size(0));

  // zeros
  auto zeros = at::zeros({2}, kInt);
  assert(kInt == zeros.options().dtype());
  assert(tensor({
    0, 0
  }, kInt).equal(zeros));

  // allclose
  auto fTensor = at::tensor({5.5f, 3.f}, kFloat);
  assert(fTensor.allclose(at::tensor({5.5, 3.0}, kFloat)));
}

void op() {
  // +
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::zeros({2, 2}); // float type
  auto c = a + b.to(at::kInt);
  assert(c.equal(a));

  // pow
  auto d = tensor({2,0}, kInt);
  d.pow(2).equal(tensor({4,0}, kInt));
  d.equal(tensor({2,0}, kInt));

  // pow modify self
  d.pow_(2);
  d.equal(tensor({4,0}, kInt));
}

void accessors() {
  at::Tensor foo = at::eye(5, kInt);

  // assert foo is 2-dimensional and holds floats.
  auto foo_a = foo.accessor<int,2>();

  for(int i = 0; i < foo_a.size(0); i++) {
    for(int j = 0; j < foo_a.size(1); j++) {
      cout << foo_a[i][j] << " ";
    }
    cout << endl;
  }
}

void from_blob() {
  int data[] = { 1, 2, 3, 4, 5, 6};
  at::Tensor f = at::from_blob(data, {2, 3}, kInt);
  cout << f << endl;
  cout << f.reshape({3,2}) << endl;
}

void index() {
  int data[] = { 1, 2, 3, 4, 5, 6};
  at::Tensor a = at::from_blob(data, {2, 3}, kInt);

  assert(a.index({0,1}).reshape({1}).equal(tensor(2, kInt)));

  // tensor[1::2]
  cout << a.index({at::indexing::Slice(1, at::indexing::None, 2)}) << endl;
}

int main() {
  create();
  op();
  accessors();
  from_blob();
  index();
}

