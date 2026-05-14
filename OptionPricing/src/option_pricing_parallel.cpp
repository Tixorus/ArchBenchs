#include <iostream>
#include <chrono>
#include <cmath>
#include "math.h"
#ifdef USE_MKL
#include "mkl.h"
#endif
#include <omp.h>
#include <cstdio>

#include "european_analytical.hpp"
#include "european_numerical.hpp"
#include "european_numerical_risc.hpp" 
#include "bermuda_RT.hpp"
#include "bermuda_SM.hpp"

void generate_correlation_matrix(int stock_count, double* mat)
{
    int ptr = 0;
    for (int i = 0; i < stock_count; i++)
    {
        mat[ptr] = 1;
        ptr++;
        for (int j = i + 1; j < stock_count; j++)
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

    long long euro_paths = 100000000;
    long long sm_paths = 1000;
    long long rt_paths = 100;

    int threads[] = { 1, 2, 4, 6, 12 };
    int num_thread_configs = 5;

    double analytical_price = calc_real_price(2.0, strike_price, interest_rate, Ar[0]);
    std::cout << "Analytical European: " << analytical_price << "\n\n";

    double base_time = 0;

    std::cout << "--- Numerical European (Single) ---\n";

    Option_EU opt_eu;
    opt_eu.exercise_time = 2.0;
    opt_eu.strike_price = strike_price;
    opt_eu.stock_count = 1;
    opt_eu.stocks = Ar;
    opt_eu.correlation = nullptr;

    for (int t = 0; t < num_thread_configs; t++)
    {
        omp_set_num_threads(threads[t]);
        int total_time = 0;
        double last_ans = 0;

        for (int r = 0; r < 3; r++)
        {
            RetVal res = monte_carlo_no_div(0, euro_paths, opt_eu, interest_rate, 12345 + r);
            total_time += res.time;
            last_ans = res.ans;
        }

        double avg_time = total_time / 3.0;
        if (t == 0) base_time = avg_time;

        std::cout << "Threads: " << threads[t]
            << " \tTime: " << avg_time << " ms"
            << " \tSpeedup: " << (base_time / avg_time) << "x"
            << " \tDiff: " << std::abs(analytical_price - last_ans) << "\n";
    }
    std::cout << "\n";

    std::cout << "--- Numerical European (Array of Options) ---\n";
    int num_options = 48;
    Option_EU* option_array = new Option_EU[num_options];
    RetVal* array_results = new RetVal[num_options];

    for (int i = 0; i < num_options; i++) {
        option_array[i].exercise_time = 2.0;
        option_array[i].strike_price = strike_price;
        option_array[i].stock_count = 1;
        option_array[i].stocks = Ar;
        option_array[i].correlation = nullptr;
    }

    base_time = 0;
    for (int t = 0; t < num_thread_configs; t++)
    {
        omp_set_num_threads(threads[t]);
        int total_time = 0;
        double last_ans = 0;

        for (int r = 0; r < 3; r++)
        {
            auto start_arr = std::chrono::high_resolution_clock::now();

            monte_carlo_eu_option_array(0, euro_paths / num_options, option_array, num_options, array_results, 0, interest_rate, 12345 + r);

            auto end_arr = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_arr - start_arr).count();

            last_ans = array_results[0].ans;
        }

        double avg_time = total_time / 3.0;
        if (t == 0) base_time = avg_time;

        std::cout << "Threads: " << threads[t]
            << " \tTime: " << avg_time << " ms"
            << " \tSpeedup: " << (base_time / avg_time) << "x"
            << " \tDiff: " << std::abs(analytical_price - last_ans) << "\n";
    }
    std::cout << "\n";

    delete[] option_array;
    delete[] array_results;

    std::cout << "--- Bermuda Stochastic Mesh ---\n";
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