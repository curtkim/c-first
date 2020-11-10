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

void index_slicing() {
  int data[] = { 1, 2, 3, 4, 5, 6};
  at::Tensor a = at::from_blob(data, {2, 3}, kInt);

  assert(a.index({0,1}).reshape({1}).equal(tensor(2, kInt)));

  // tensor[1::2]
  cout << a.index({at::indexing::Slice(1, at::indexing::None, 2)}) << endl;
}

// https://pytorch.org/cppdocs/notes/tensor_indexing.html
void index_slicing0() {
  int data[] = { 1, 2, 3};
  at::Tensor a = at::from_blob(data, {3}, kInt);
  // a[0:2]
  assert(
    a.index({at::indexing::Slice(0, 2)}).equal(
      tensor({1,2}, kInt)
    )
  );

  int data2[] = { 1, 2, 3,
                  4, 5, 6};
  at::Tensor b = at::from_blob(data2, {2, 3}, kInt);
  // b[0:2,1]
  assert(
    b.index({at::indexing::Slice(0, 2), 1}).equal(
      tensor({2,5}, kInt)
    )
  );

  // b[:1]
  assert(
    b.index({at::indexing::Slice(at::indexing::None, 1)}).equal(
      tensor({1,2,3}, kInt).reshape({1,3})
    )
  );

  auto c = at::arange(12, kInt).reshape({2,2,3});
  // c[1,...]
  assert(
    c.index({1, at::indexing::Ellipsis}).equal(
      tensor({6,7,8,
              9,10,11}, kInt).reshape({2,3})
    )
  );

  assert(
    c.index({0, at::indexing::Ellipsis, at::indexing::Slice(at::indexing::None, -1)}).equal(
      tensor({0,1,
              3,4}, kInt).reshape({2,2})
    )
  );
}

void max() {
  int data2[] = { 1, 2, 3,
                  4, 5, 6};
  at::Tensor b = at::from_blob(data2, {2, 3}, kInt);
  const auto [values, indices] = b.max(-1);
  assert( values.equal(tensor({3,6}, kInt)) );
  assert( indices.equal(tensor({2,2}, kLong)) );
}

int main() {
  create();
  op();
  accessors();
  from_blob();
  index_slicing();
  index_slicing0();
  max();
}

