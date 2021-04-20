#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <iterator>
#include <iostream>

int do_constant_iterator()
{
    thrust::device_vector<int> data(4);
    data[0] = 3;
    data[1] = 7;
    data[2] = 2;
    data[3] = 5;

    // add 10 to all values in data
    thrust::transform(data.begin(), data.end(),
                      thrust::constant_iterator<int>(10),
                      data.begin(),
                      thrust::plus<int>());

    // data is now [13, 17, 12, 15]

    // print result
    thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}

void do_counting_iterator() {
    // this example computes indices for all the nonzero values in a sequence

    // sequence of zero and nonzero values
    thrust::device_vector<int> stencil(8);
    stencil[0] = 0;
    stencil[1] = 1;
    stencil[2] = 1;
    stencil[3] = 0;
    stencil[4] = 0;
    stencil[5] = 1;
    stencil[6] = 0;
    stencil[7] = 1;

    // storage for the nonzero indices
    thrust::device_vector<int> indices(8);

    // counting iterators define a sequence [0, 8)
    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + 8;

    // compute indices of nonzero elements
    typedef thrust::device_vector<int>::iterator IndexIterator;

    IndexIterator indices_end = thrust::copy_if(first, last,
                                                stencil.begin(),
                                                indices.begin(),
                                                thrust::identity<int>());
    // indices now contains [1,2,5,7]

    // print result
    std::cout << "found " << (indices_end - indices.begin()) << " nonzero values at indices:\n";
    thrust::copy(indices.begin(), indices_end, std::ostream_iterator<int>(std::cout, "\n"));

}

int main(void){
    do_constant_iterator();
    do_counting_iterator();
}