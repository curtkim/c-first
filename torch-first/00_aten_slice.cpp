#include <ATen/ATen.h>
#include <iostream>

using namespace std;
using namespace at;
using namespace at::indexing;

// 아래 두개를 구분
// at::indexing::Ellipsis
// at::indexing::None

void index_slicing() {
    at::Tensor a = at::tensor({1, 2, 3,
                               4, 5, 6}).reshape({2, 3});

    // a[0,1]
    assert(a.index({0, 1}).reshape({1}).equal(tensor(2, kInt)));

    // tensor[, 1::2] [모든row, 한칸씩 건너뛰고]
    assert(
            a.index({Ellipsis, Slice(0, None, 2)}).equal(
                    tensor({1, 3, 4, 6}).reshape({2, 2})
            )
    );
}

void index_slicing_dim1() {
    at::Tensor a = tensor({1, 2, 3});

    // a[0:2]
    assert(
            a.index({Slice(0, 2)}).equal(
                    tensor({1, 2}, kInt)
            )
    );

    // two step a[0::2]
    assert(
            a.index({Slice(0, None, 2)}).equal(
                    tensor({1, 3}, kInt)
            )
    );

}

// https://pytorch.org/cppdocs/notes/tensor_indexing.html
void index_slicing_dim2() {

    at::Tensor b = tensor({1, 2, 3, 4, 5, 6}).reshape({2, 3});

    // b[0:2,1]
    assert(
            b.index({Slice(0, 2), 1}).equal(
                    tensor({2, 5})
            )
    );

    // b[:1,]
    assert(
            b.index({Slice(None, 1)}).equal(
                    tensor({1, 2, 3}).reshape({1, 3})
            )
    );

    // b[,:1]
    assert(
            b.index({Ellipsis, Slice(None, 1)}).equal(
                    tensor({1, 4}).reshape({2, 1})
            )
    );

}

void index_slicing_dim3() {
    auto c = at::arange(12, kInt).reshape({2, 2, 3});

    // c[1,...]
    assert(
            c.index({1, Ellipsis}).equal(
                    tensor({6, 7, 8,
                            9, 10, 11}).reshape({2, 3})
            )
    );

    assert(
            c.index({0, Ellipsis, Slice(None, -1)}).equal(
                    tensor({0, 1,
                            3, 4}).reshape({2, 2})
            )
    );

    auto d = at::arange(12, kInt).reshape({1, 3, 4});
    auto d2 = at::arange(8, kInt).reshape({1, 2, 4});
    assert(d.index({Slice(0, 1), Slice(0, 2), Ellipsis}).equal(d2));

    auto e = at::arange(12, kInt).reshape({1, 3, 4});
    assert(
            e.index({Slice(0, 1), Slice(0, 2), Ellipsis}).equal(
                    tensor({0, 1, 2, 3,
                            4, 5, 6, 7}).reshape({1, 2, 4})
            )
    );

}

int main() {
    index_slicing();
    index_slicing_dim1();
    index_slicing_dim2();
    index_slicing_dim3();
}

