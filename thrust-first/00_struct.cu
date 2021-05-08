#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

struct COLOR {
    uint8_t R;
    uint8_t G;
    uint8_t B;
};

std::ostream &operator<<(std::ostream &os, COLOR const &m) {
    return os << m.R << " " << m.G << " " << m.B;
}

int main(void)
{
    const int height = 1;
    const int width = 2;

    COLOR RED = {255, 0, 0};
    std::cout << RED << std::endl;

    std::vector<COLOR> A(height*width);
    std::fill(A.begin(), A.end(), RED);
    for(int i = 0; i < A.size(); i++)
        std::cout << "A[" << i << "] = " << A[i] << std::endl;


    thrust::device_vector<COLOR> D(height*width);
    thrust::fill(D.begin(), D.end(), RED);

    thrust::host_vector<COLOR> H(height*width);
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for(int i = 0; i < D.size(); i++)
        std::cout << "D[" << i << "] = " << D[i] << std::endl;
    for(int i = 0; i < H.size(); i++)
        std::cout << "H[" << i << "] = " << H[i] << std::endl;

}
