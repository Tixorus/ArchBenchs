#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>

#include "european_analytical.hpp"
#include "european_numerical.hpp"
#include "bermuda_RT.hpp"
#include "bermuda_SM.hpp"

void generate_correlation_matrix(int stock_count, double* mat)
{
    int ptr = 0;
    for (int i = 0; i < stock_count; i++)
    {
        mat[ptr] = 1;
        ptr++;
        for (int j = i+1; j < stock_count; j++)
        {
            mat[ptr] = 0.5;
            ptr++;
        }
    }
}

int main()
{
    Stock Ar[5];
    for (int i = 0; i < 5; i++)
    {
        Ar[i].S0 = 200;
        Ar[i].volatility = 0.5;
        Ar[i].dividends = 0.0;
    }

    double strike_price = 200.0;
    double interest_rate = 0.07;
    double tim[] = { 0.5, 1.0, 1.5, 2.0 };
    int exercise_num = 4;

    long long euro_paths = 200000000;
    long long sm_paths = 3000;
    long long rt_paths = 100;

    int threads[] = { 1, 2, 4, 6, 12 };
    int num_thread_configs = 5;

    std::cout << "Analytical European: "
        << calc_real_price(2.0, strike_price, interest_rate, Ar[0]) << "\n\n";

    double base_time = 0;

    std::cout << "Numerical European \n";
    for (int t = 0; t < num_thread_configs; t++)
    {
        omp_set_num_threads(threads[t]);
        int total_time = 0;

        for (int r = 0; r < 3; r++)
        {
            RetVal res = monte_carlo_no_div(0, euro_paths, 2.0, strike_price, interest_rate, 1, Ar, 12345 + r);
            total_time += res.time;
        }

        double avg_time = total_time / 3.0;
        if (t == 0) base_time = avg_time; 

        std::cout << "Threads: " << threads[t]
            << " \tTime: " << avg_time << " ms"
            << " \tSpeedup: " << (base_time / avg_time) << "x\n";
    }
    std::cout << "\n";

  
    std::cout << "Bermuda Stochastic Mesh\n";
    base_time = 0;
    for (int t = 0; t < num_thread_configs; t++)
    {
        omp_set_num_threads(threads[t]);
        int total_time = 0;

        for (int r = 0; r < 3; r++)
        {
            RetVal res = monte_carlo_bermuda_SM(0, sm_paths, strike_price, interest_rate, 1, Ar, exercise_num, tim);
            total_time += res.time;
        }

        double avg_time = total_time / 3.0;
        if (t == 0) base_time = avg_time;

        std::cout << "Threads: " << threads[t]
            << " \tTime: " << avg_time << " ms"
            << " \tSpeedup: " << (base_time / avg_time) << "x\n";
    }
    std::cout << "\n";


    std::cout << "--- Bermuda RT Recursive Parallel ---\n";
    base_time = 0;
    for (int t = 0; t < num_thread_configs; t++)
    {
        omp_set_num_threads(threads[t]);
        int total_time = 0;

        for (int r = 0; r < 3; r++)
        {
            RetVal res = monte_carlo_bermuda_RT_recursive_parallel(0, rt_paths, strike_price, interest_rate, 1, Ar, exercise_num, tim);
            total_time += res.time;
        }

        double avg_time = total_time / 3.0;
        if (t == 0) base_time = avg_time;

        std::cout << "Threads: " << threads[t]
            << " \tTime: " << avg_time << " ms"
            << " \tSpeedup: " << (base_time / avg_time) << "x\n";
    }

    return 0;
}