#include <nppi.h>
#include <iostream>

const int w = 16;
const int h = 16;
const int B = 0;
const int G = 1;
const int R = 2;
const int A = 3;

int main(){

    Npp8u *pSrc, *hSrc, *pDst[3];
    int nSrcStep, rDstStep[3];
    NppiSize oSizeROI{w, h};

    // create test packed BGRA image
    hSrc = new Npp8u[h*w*4];
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w/4; j++){
            // red bar
            hSrc[(i*w+j)*4+B] = 0;
            hSrc[(i*w+j)*4+G] = 0;
            hSrc[(i*w+j)*4+R] = 255;
            hSrc[(i*w+j)*4+A] = 255;
            // green bar
            hSrc[(i*w+j+(w/4))*4+B] = 0;
            hSrc[(i*w+j+(w/4))*4+G] = 255;
            hSrc[(i*w+j+(w/4))*4+R] = 0;
            hSrc[(i*w+j+(w/4))*4+A] = 255;
            // blue bar
            hSrc[(i*w+j+(w/2))*4+B] = 255;
            hSrc[(i*w+j+(w/2))*4+G] = 0;
            hSrc[(i*w+j+(w/2))*4+R] = 0;
            hSrc[(i*w+j+(w/2))*4+A] = 255;
            // white bar
            hSrc[(i*w+j+(3*w/4))*4+B] = 255;
            hSrc[(i*w+j+(3*w/4))*4+G] = 255;
            hSrc[(i*w+j+(3*w/4))*4+R] = 255;
            hSrc[(i*w+j+(3*w/4))*4+A] = 255;
        }


    cudaMalloc(&pSrc, h*w*4*sizeof(Npp8u));
    cudaMemcpy(pSrc, hSrc, h*w*4*sizeof(Npp8u), cudaMemcpyHostToDevice);

    nSrcStep = w*4*sizeof(Npp8u);

    cudaMalloc(pDst+0, h*w*sizeof(Npp8u));     // Y storage
    cudaMalloc(pDst+1, (h*w*sizeof(Npp8u))/4); // U storage
    cudaMalloc(pDst+2, (h*w*sizeof(Npp8u))/4); // V storage

    cudaMemset(pDst+0, 0,  h*w*sizeof(Npp8u));
    cudaMemset(pDst+1, 0, (h*w*sizeof(Npp8u))/4);
    cudaMemset(pDst+2, 0, (h*w*sizeof(Npp8u))/4);

    rDstStep[0] = w*sizeof(Npp8u);
    rDstStep[1] = (w/2)*sizeof(Npp8u);
    rDstStep[2] = (w/2)*sizeof(Npp8u);

    NppStatus stat = nppiBGRToYUV420_8u_AC4P3R(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
    if (stat != NPP_SUCCESS)
        std::cout << "NPP error: " << (int)stat << std::endl;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    Npp8u *hY, *hU, *hV;
    hY = new Npp8u[h*w];
    hU = new Npp8u[h*w/4];
    hV = new Npp8u[h*w/4];
    cudaMemcpy(hY, pDst[0], h*w*sizeof(Npp8u),     cudaMemcpyDeviceToHost);
    cudaMemcpy(hU, pDst[1], h*(w/4)*sizeof(Npp8u), cudaMemcpyDeviceToHost);
    cudaMemcpy(hV, pDst[2], h*(w/4)*sizeof(Npp8u), cudaMemcpyDeviceToHost);

    // from https://en.wikipedia.org/wiki/YUV
    std::cout << "Expected values: " << std::endl;
    std::cout << "color  Y   U   V" << std::endl;

    int Yred = 0.299*255;
    int Ured = (-0.147*255)+128;
    int Vred = (0.615*255)+128;
    if (Yred > 255) Yred = 255;
    if (Ured > 255) Ured = 255;
    if (Vred > 255) Vred = 255;
    if (Yred < 0) Yred = 0;
    if (Ured < 0) Ured = 0;
    if (Vred < 0) Vred = 0;
    std::cout << "RED:   " << Yred << " " << Ured << " " << Vred << std::endl;

    int Ygrn = 0.587*255;
    int Ugrn = (-0.289*255)+128;
    int Vgrn = (-0.515*255)+128;
    if (Ygrn > 255) Ygrn = 255;
    if (Ugrn > 255) Ugrn = 255;
    if (Vgrn > 255) Vgrn = 255;
    if (Ygrn < 0) Ygrn = 0;
    if (Ugrn < 0) Ugrn = 0;
    if (Vgrn < 0) Vgrn = 0;
    std::cout << "GREEN: " << Ygrn << " " << Ugrn << " " << Vgrn << std::endl;

    int Yblu = 0.114*255;
    int Ublu = (0.436*255)+128;
    int Vblu = (-0.100*255)+128;
    if (Yblu > 255) Yblu = 255;
    if (Ublu > 255) Ublu = 255;
    if (Vblu > 255) Vblu = 255;
    if (Yblu < 0) Yblu = 0;
    if (Ublu < 0) Ublu = 0;
    if (Vblu < 0) Vblu = 0;
    std::cout << "BLUE:  " << Yblu << " " << Ublu << " " << Vblu << std::endl;


    std::cout << "Y plane:" << std::endl;
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++)
            std::cout << (unsigned)hY[i*w+j] <<  " ";
        std::cout << std::endl;}

    std::cout << "U plane:" << std::endl;
    for (int i = 0; i < h/2; i++){
        for (int j = 0; j < w/2; j++)
            std::cout << (unsigned)hU[i*(w/2)+j] <<  " ";
        std::cout << std::endl;}

    std::cout << "V plane:" << std::endl;
    for (int i = 0; i < h/2; i++){
        for (int j = 0; j < w/2; j++)
            std::cout << (unsigned)hV[i*(w/2)+j] <<  " ";
        std::cout << std::endl;}
}