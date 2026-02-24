#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>

#include "numerical.hpp"

//generator values:
//0 == VSL_BRNG_MCG59;
//1 == VSL_BRNG_SOBOL;
//2 == VSL_BRNG_NIEDERR;
//3 == VSL_BRNG_MT19937
double monte_carlo_cont_div(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int seed)
{
    double C = 0;
    int buffer = 50000 / stock_count;

    double timesq = sqrt(time);
    double drift[50], sigma_sqrt[50], s0[50], divexp[50];
    for (int j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - stocks[j].dividends - 0.5 * stocks[j].volatility * stocks[j].volatility) * time;
        sigma_sqrt[j] = stocks[j].volatility * timesq;
        s0[j] = stocks[j].S0;
        divexp[j] = exp((-interest_rate + stocks[j].dividends) * time);
    }

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        long long thread_id = omp_get_thread_num();
        double* gauss = new double[buffer * stock_count];
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 2)
        {
            vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 3)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }

#pragma omp for reduction(+:C) schedule(static)
        for (int block = 0; block < paths / buffer; block++)
        {
            double Cbuff = 0;
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
            for (int i = 0; i < buffer; i++)
            {
                double max1 = 0.0, max2 = 0.0;
                int max1div = 0, max2div = 0;
                for (int j = 0; j < stock_count; j++)
                {
                    double d = sigma_sqrt[j] * gauss[j + i * stock_count];
                    double e = exp(drift[j] + d);
                    double r = s0[j] * e;
                    if (r > max1) { max1 = r; max1div = j; }
                    e = exp(drift[j] - d);
                    r = s0[j] * e;
                    if (r > max2) { max2 = r; max2div = j; }
                }

                Cbuff += fmax(0.0, (max1 - strike_price)) * divexp[max1div] / 2;
                Cbuff += fmax(0.0, (max2 - strike_price)) * divexp[max2div] / 2;
            }
            C += Cbuff / paths;
        }
        vslDeleteStream(&stream);
        delete[] gauss;
    }//parallel end

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    //std::cout << "Parallel " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n"; 
    std::cout << duration_ms.count() << "\n";

    return C;
}

//generator values:
//0 == VSL_BRNG_MCG59;
//1 == VSL_BRNG_SOBOL;
//2 == VSL_BRNG_NIEDERR;
//3 == VSL_BRNG_MT19937
double monte_carlo_no_div(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int seed)
{
    double C = 0;
    int buffer = 50000 / stock_count;

    double timesq = sqrt(time);
    double drift[50], sigma_sqrt[50], s0[50];
    for (int j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - stocks[j].dividends - 0.5 * stocks[j].volatility * stocks[j].volatility) * time;
        sigma_sqrt[j] = stocks[j].volatility * timesq;
        s0[j] = stocks[j].S0;
    }

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        long long thread_id = omp_get_thread_num();
        double* gauss = new double[buffer * stock_count];
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 2)
        {
            vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 3)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }

#pragma omp for reduction(+:C) schedule(static)
        for (int block = 0; block < paths / buffer; block++)
        {
            double Cbuff = 0;
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
            for (int i = 0; i < buffer; i++)
            {
                double max1 = 0.0, max2 = 0.0;
                for (int j = 0; j < stock_count; j++)
                {
                    double d = sigma_sqrt[j] * gauss[j + i * stock_count];
                    double e = exp(drift[j] + d);
                    double r = s0[j] * e;
                    if (r > max1) { max1 = r; }
                    e = exp(drift[j] - d);
                    r = s0[j] * e;
                    if (r > max2) { max2 = r; }
                }
                Cbuff += fmax(0.0, (max1 - strike_price));
                Cbuff += fmax(0.0, (max2 - strike_price));
            }
            C += Cbuff / paths;
        }
        vslDeleteStream(&stream);
        delete[] gauss;
    }//parallel end

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    //std::cout << "Parallel " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n"; 
    std::cout << duration_ms.count() << "\n";

    return C * exp(-interest_rate * time) / 2;
}

double monte_carlo_multivariate(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, double* correlation, int seed)
{
    double C = 0;
    int buffer = 50000 / stock_count;

    double timesq = sqrt(time);
    double drift[50], sigma_sqrt[50], s0[50];
    for (int j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - stocks[j].dividends - 0.5 * stocks[j].volatility * stocks[j].volatility) * time;
        sigma_sqrt[j] = stocks[j].volatility * timesq;
        s0[j] = stocks[j].S0;
    }
    double gmean[50];
    for (int j = 0; j < stock_count; j++)
    {
        gmean[j] = 0.0;
    }
    

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        long long thread_id = omp_get_thread_num();
        double* gauss = new double[buffer * stock_count];
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 2)
        {
            vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 3)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, seed);
            vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
        }

#pragma omp for reduction(+:C) schedule(static)
        for (int block = 0; block < paths / buffer; block++)
        {
            double Cbuff = 0;
            vdRngGaussianMV(VSL_RNG_METHOD_GAUSSIANMV_ICDF, stream, buffer, gauss, stock_count, VSL_MATRIX_STORAGE_PACKED, (const double*)gmean, (const double*)correlation);
            for (int i = 0; i < buffer; i++)
            {
                double max1 = 0.0, max2 = 0.0;
                for (int j = 0; j < stock_count; j++)
                {
                    double d = sigma_sqrt[j] * gauss[j + i * stock_count];
                    double e = exp(drift[j] + d);
                    double r = s0[j] * e;
                    if (r > max1) { max1 = r; }
                    e = exp(drift[j] - d);
                    r = s0[j] * e;
                    if (r > max2) { max2 = r; }
                }
                Cbuff += fmax(0.0, (max1 - strike_price));
                Cbuff += fmax(0.0, (max2 - strike_price));
            }
            C += Cbuff / paths;
        }
        vslDeleteStream(&stream);
        delete[] gauss;
    }//parallel end

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    //std::cout << "Parallel " << (quasi ? "Quasi" : "Pseudo") << " MC finished in: " << duration_ms.count() << " milliseconds\n"; 
    std::cout << duration_ms.count() << "\n";

    return C * exp(-interest_rate * time)/2;
}