#include <iostream>
#include <cufft.h>
#include <vector>
#include <chrono>

#include <fftw3.h>
#include <thread>

constexpr int NX = 256;
constexpr int NY = 256;
constexpr int RANK = 2;
constexpr int BATCHES = 16*32;
constexpr int SLICE_SIZE = NX * NY;
constexpr size_t BYTES = size_t(NX) * size_t(NY) * size_t(BATCHES)*sizeof(cufftComplex);

void runFFT(cufftComplex *input, cufftComplex *output)
{

    std::vector<int> n = {NX, NY};
    cufftHandle plan;
    if (cufftPlanMany(&plan, RANK, n.data(), nullptr, 1, SLICE_SIZE, nullptr, 1, SLICE_SIZE, cufftType::CUFFT_C2C, BATCHES) != CUFFT_SUCCESS)
    {
        //if (cufftPlanMany(&plan, RANK, n.data(), n.data(), 1,SLICE_SIZE,n.data(),1,SLICE_SIZE,cufftType::CUFFT_C2C,BATCHES) != CUFFT_SUCCESS){
        std::cout << "CUFFT Plan many failed " << std::endl;
    }

    cufftExecC2C(plan, input, output, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    if (cufftDestroy(plan) != CUFFT_SUCCESS)
    {
        std::cout << "Faield destroying plan" << std::endl;
    }
}

void runFFTW(fftwf_complex *in, fftwf_complex *out)
{

    std::vector<int> n = {NX, NY};
    auto plan = fftwf_plan_many_dft(2, n.data(),BATCHES, in,nullptr, 1,SLICE_SIZE, out, nullptr,1,SLICE_SIZE,1, FFTW_ESTIMATE);

    fftwf_execute_dft(plan, in, out);
    fftwf_destroy_plan(plan);
}

int main(int, char **)
{
    std::cout << "Hello, world!\n";


    {
        cufftComplex *input;
        cufftComplex *output;
        cudaMalloc(&input, BYTES);
        cudaMalloc(&output, BYTES);

        auto start = std::chrono::high_resolution_clock::now();

        constexpr int repetitions = 10;

        for (int i = 0; i < repetitions; i++)
        {
            runFFT(input, output);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Time per FFT " << elapsed_seconds.count() / repetitions << "s" << std::endl;

        cudaFree(input);
        cudaFree(output);
    }

    {
        fftwf_complex* input = (fftwf_complex*)malloc(BYTES);
        fftwf_complex* output = (fftwf_complex*)malloc(BYTES);
        
        constexpr int repetitions = 10;
        fftwf_init_threads();
        fftwf_plan_with_nthreads(std::thread::hardware_concurrency());

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < repetitions; i++)
        {
            runFFTW(input, output);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "Time per FFT " << elapsed_seconds.count() / repetitions << "s" << std::endl;
        free(input);
        free(output);

    }

}