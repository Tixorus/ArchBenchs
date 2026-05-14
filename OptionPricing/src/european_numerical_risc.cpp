#include <iostream>
#include <chrono>
#include "math.h"
#ifdef USE_MKL
#include "mkl.h"
#endif
#include <cstdio>
#include <omp.h>

#include "european_numerical_risc.hpp"

void monte_carlo_eu_option_array(int generator, long long paths_per_option, Option_EU* options, int num_options, RetVal* results, int option_type, double interest_rate, uint64_t seed)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_options; i++)
    {
        int current_seed = seed + i;
        if (option_type == 0)
        {
            results[i] = monte_carlo_no_div_linear(generator, paths_per_option, options[i], interest_rate, current_seed);
        }
        else if (option_type == 1)
        {
            results[i] = monte_carlo_cont_div_linear(generator, paths_per_option, options[i], interest_rate, current_seed);
        }
        /*else if (option_type == 2)
        {
            results[i] = monte_carlo_multivariate_linear(generator, paths_per_option, options[i], interest_rate, current_seed);
        }*/ //TODO
    }

}

RetVal monte_carlo_cont_div_linear(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed)
{
    double time = option.exercise_time;
    double strike_price = option.strike_price;
    unsigned int stock_count = option.stock_count;
    Stock* stocks = option.stocks;

    double C = 0;
    int buffer = 50000 / stock_count;

    double timesq = sqrt(time);
    double drift[50], sigma_sqrt[50], s0[50], divexp[50];
    for (int j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - stocks[j].dividends - 0.5 * stocks[j].volatility * stocks[j].volatility) * time;
        sigma_sqrt[j] = stocks[j].volatility * timesq;
        s0[j] = stocks[j].S0;
        divexp[j] = CUSTOM_EXP((-interest_rate + stocks[j].dividends) * time);
    }

    auto start = std::chrono::high_resolution_clock::now();

#ifdef USE_MKL
    VSLStreamStatePtr stream;
    double* gauss = new double[buffer * stock_count];

    if (generator == 0) vslNewStream(&stream, VSL_BRNG_MCG59, seed);
    else if (generator == 1) vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
    else if (generator == 2) vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
    else if (generator == 3) vslNewStream(&stream, VSL_BRNG_MT19937, seed);

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
                double e = CUSTOM_EXP(drift[j] + d);
                double r = s0[j] * e;
                if (r > max1) { max1 = r; max1div = j; }
                e = CUSTOM_EXP(drift[j] - d);
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
#else
    uint32_t thread_id = omp_get_thread_num();
    openrand::Philox rng(seed, thread_id);
    bool has_spare = false;
    double spare_gauss = 0.0;

    for (long long i = 0; i < paths; i++)
    {
        double max1 = 0.0, max2 = 0.0;
        int max1div = 0, max2div = 0;
        for (int j = 0; j < stock_count; j++)
        {
            double gauss_val;
            if (has_spare)
            {
                gauss_val = spare_gauss;
                has_spare = false;
            }
            else
            {
                double u1, u2;
                generate_uniform_pair(rng, u1, u2);
                double radius = sqrt(-2.0 * log(u1));
                double theta = 2.0 * 3.14159265358979323846 * u2;
                gauss_val = radius * CUSTOM_COS(theta);
                spare_gauss = radius * CUSTOM_SIN(theta);
                has_spare = true;
            }

            double d = sigma_sqrt[j] * gauss_val;
            double e = CUSTOM_EXP(drift[j] + d);
            double r = s0[j] * e;
            if (r > max1) { max1 = r; max1div = j; }

            e = CUSTOM_EXP(drift[j] - d);
            r = s0[j] * e;
            if (r > max2) { max2 = r; max2div = j; }
        }
        double path_payoff = fmax(0.0, (max1 - strike_price)) * divexp[max1div] / 2 + fmax(0.0, (max2 - strike_price)) * divexp[max2div] / 2;
        C += path_payoff / paths;
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    RetVal ret;
    ret.ans = C;
    ret.time = duration_ms.count();

    return ret;
}

RetVal monte_carlo_no_div_linear(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed)
{
    double time = option.exercise_time;
    double strike_price = option.strike_price;
    unsigned int stock_count = option.stock_count;
    Stock* stocks = option.stocks;

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

#ifdef USE_MKL
    VSLStreamStatePtr stream;
    double* gauss = new double[buffer * stock_count];

    if (generator == 0) vslNewStream(&stream, VSL_BRNG_MCG59, seed);
    else if (generator == 1) vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
    else if (generator == 2) vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
    else if (generator == 3) vslNewStream(&stream, VSL_BRNG_MT19937, seed);

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
                double e = CUSTOM_EXP(drift[j] + d);
                double r = s0[j] * e;
                if (r > max1) { max1 = r; }
                e = CUSTOM_EXP(drift[j] - d);
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
#else
    uint32_t thread_id = omp_get_thread_num();
    openrand::Philox rng(seed, thread_id);
    bool has_spare = false;
    double spare_gauss = 0.0;

    for (long long i = 0; i < paths; i++)
    {
        double max1 = 0.0, max2 = 0.0;
        for (int j = 0; j < stock_count; j++)
        {
            double gauss_val;
            if (has_spare)
            {
                gauss_val = spare_gauss;
                has_spare = false;
            }
            else
            {
                double u1, u2;
                generate_uniform_pair(rng, u1, u2);
                double radius = sqrt(-2.0 * log(u1));
                double theta = 2.0 * 3.14159265358979323846 * u2;
                gauss_val = radius * CUSTOM_COS(theta);
                spare_gauss = radius * CUSTOM_SIN(theta);
                has_spare = true;
            }

            double d = sigma_sqrt[j] * gauss_val;
            double e = CUSTOM_EXP(drift[j] + d);
            double r = s0[j] * e;
            if (r > max1) { max1 = r; }

            e = CUSTOM_EXP(drift[j] - d);
            r = s0[j] * e;
            if (r > max2) { max2 = r; }
        }
        double path_payoff = fmax(0.0, (max1 - strike_price)) + fmax(0.0, (max2 - strike_price));
        C += path_payoff / paths;
    }
#endif

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    RetVal ret;
    ret.ans = C * CUSTOM_EXP(-interest_rate * time) / 2;
    ret.time = duration_ms.count();

    return ret;
}