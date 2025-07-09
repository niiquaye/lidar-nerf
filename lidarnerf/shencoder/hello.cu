#include <iostream>
__global__ void hello() {
    printf("Hello from CUDA!\n");
}
int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

