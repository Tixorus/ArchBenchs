#include "bermuda.hpp"
#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>
#include <stack>

double monte_carlo_bermuda_RT_linear(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    double prevtime = 0;
    double* drift = new double[(exercise_num+1)*stock_count];
    double* discount = new double[exercise_num+1];
    double* timesq = new double[exercise_num+1];
    double* high_avg = new double[exercise_num+1];
    int* ptr = new int[exercise_num + 1];
    double* low_avg = new double[paths*(exercise_num+1)];

    high_avg[0] = 0;
    ptr[0] = 0;

    for (int i = 1; i <= exercise_num; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            drift[i * stock_count + j] = exp((interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * (exercise_dates[i-1] - prevtime));
        }
        discount[i] = exp(-interest_rate * (exercise_dates[i-1] - prevtime));
        timesq[i] = sqrt(exercise_dates[i-1] - prevtime);
        prevtime = exercise_dates[i-1];
        high_avg[i] = 0;
        ptr[i] = 0;
    }

	auto start = std::chrono::high_resolution_clock::now();

    VSLStreamStatePtr stream;
    long long thread_id = omp_get_thread_num();
    double* gauss = new double[stock_count];
    

    if (generator == 0)
    {
        vslNewStream(&stream, VSL_BRNG_MCG59, 1234567);
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
        vslNewStream(&stream, VSL_BRNG_MT19937, 1234567);
        vslSkipAheadStream(stream, thread_id * (paths * stock_count / omp_get_num_threads() + 1) + 1);
    }
    std::stack<int> s;
    double* prices_ar = new double[stock_count * (exercise_num+1)];
    int prev = 0;
    s.push(0);

    for (int i = 0; i < stock_count; i++) prices_ar[i] = stocks[i].S0;
    for (int i = 0; i < paths; i++) s.push(1);
    while (!s.empty())
    {
        int cur = s.top();
        if (cur >= prev)
        {
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, stock_count, gauss, 0.0, 1.0);
            for (int i = 0; i < stock_count; i++) prices_ar[cur * stock_count + i] = prices_ar[(cur - 1) * stock_count + i] * drift[cur * stock_count + i] * exp(stocks[i].volatility * timesq[cur] * gauss[i]);

            if (cur == exercise_num)
            {
                double mx = 0;
                for (int i = 0; i < stock_count; i++) mx = std::max(mx, prices_ar[cur * stock_count + i]);
                high_avg[cur] += std::max(mx - strike_price, 0.0);
                low_avg[cur*paths+(ptr[cur])%paths] = std::max(mx - strike_price, 0.0);
                ptr[cur]++;
                s.pop();
            }
            else
            {
                for (int i = 0; i < paths; i++) s.push(cur + 1);
            }
        }
        else
        {
            double mx = 0;
            for (int i = 0; i < stock_count; i++) mx = std::max(mx, prices_ar[cur * stock_count + i]);
            mx -= strike_price;
            high_avg[cur] += std::max(mx, (high_avg[cur + 1] * discount[cur + 1])/paths);
            high_avg[cur + 1] = 0;
            double low_sum = 0;
            double low_est = 0;
            for (int i = 0; i < paths; i++) low_sum += low_avg[(cur + 1) * paths + i];
            for (int i = 0; i < paths; i++)
            {
                low_sum -= low_avg[(cur + 1) * paths + i];
                if (mx >= (low_sum / (paths - 1)) * discount[cur + 1]) low_est += mx;
                else low_est += (low_avg[(cur + 1) * paths + i]) * discount[cur + 1];
                low_sum += low_avg[(cur + 1) * paths + i];
            }
            low_avg[cur * paths + (ptr[cur]) % paths] = low_est/paths;
            ptr[cur]++;
            s.pop();
        }
        prev = cur;
    }
    std::cout << low_avg[0] << "\n" << high_avg[0];
    return 0;
}

double monte_carlo_bermuda_RT_parallel(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    double prevtime = 0;
    double* drift = new double[(exercise_num + 1) * stock_count];
    double* discount = new double[exercise_num + 1];
    double* timesq = new double[exercise_num + 1];

    for (int i = 1; i <= exercise_num; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            drift[i * stock_count + j] = exp((interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * (exercise_dates[i - 1] - prevtime));
        }
        discount[i] = exp(-interest_rate * (exercise_dates[i - 1] - prevtime));
        timesq[i] = sqrt(exercise_dates[i - 1] - prevtime);
        prevtime = exercise_dates[i - 1];
    }

    auto start = std::chrono::high_resolution_clock::now();
    double low_ans = 0, high_ans = 0;
    
#pragma omp parallel reduction(+:low_ans, high_ans)
    {
        VSLStreamStatePtr stream;
        long long thread_id = omp_get_thread_num();
        double* gauss = new double[stock_count];
        
        double* high_avg = new double[exercise_num + 1];
        int* ptr = new int[exercise_num + 1];
        double* low_avg = new double[paths * (exercise_num + 1)];
        unsigned long long points_num = stock_count;
        high_avg[0] = 0;
        ptr[0] = 0;
        for (int i = 1; i <= exercise_num; i++)
        {
            high_avg[i] = 0;
            ptr[i] = 0;
            points_num *= paths;
        }
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, 1234567);
            vslSkipAheadStream(stream, thread_id * (points_num + 1) + 1);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, thread_id * (points_num + 1) + 1);
        }
        else if (generator == 2)
        {
            vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
            vslSkipAheadStream(stream, thread_id * (points_num + 1) + 1);
        }
        else if (generator == 3)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, 1234567);
            vslSkipAheadStream(stream, thread_id * (points_num + 1) + 1);
        }
        std::stack<int> s;
        double* prices_ar = new double[stock_count * (exercise_num + 1)];
        int prev = 0;
        s.push(0);

        for (int i = 0; i < stock_count; i++) prices_ar[i] = stocks[i].S0;
        for (int i = 0; i < paths; i++) s.push(1);
        while (!s.empty())
        {
            int cur = s.top();
            if (cur >= prev)
            {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, stock_count, gauss, 0.0, 1.0);
                for (int i = 0; i < stock_count; i++) prices_ar[cur * stock_count + i] = prices_ar[(cur - 1) * stock_count + i] * drift[cur * stock_count + i] * exp(stocks[i].volatility * timesq[cur] * gauss[i]);

                if (cur == exercise_num)
                {
                    double mx = 0;
                    for (int i = 0; i < stock_count; i++) mx = std::max(mx, prices_ar[cur * stock_count + i]);
                    high_avg[cur] += std::max(mx - strike_price, 0.0);
                    low_avg[cur * paths + (ptr[cur]) % paths] = std::max(mx - strike_price, 0.0);
                    ptr[cur]++;
                    s.pop();
                }
                else
                {
                    for (int i = 0; i < paths; i++) s.push(cur + 1);
                }
            }
            else
            {
                double mx = 0;
                for (int i = 0; i < stock_count; i++) mx = std::max(mx, prices_ar[cur * stock_count + i]);
                mx -= strike_price;
                high_avg[cur] += std::max(mx, (high_avg[cur + 1] * discount[cur + 1]) / paths);
                high_avg[cur + 1] = 0;
                double low_sum = 0;
                double low_est = 0;
                for (int i = 0; i < paths; i++) low_sum += low_avg[(cur + 1) * paths + i];
                for (int i = 0; i < paths; i++)
                {
                    low_sum -= low_avg[(cur + 1) * paths + i];
                    if (mx >= (low_sum / (paths - 1)) * discount[cur + 1]) low_est += mx;
                    else low_est += (low_avg[(cur + 1) * paths + i]) * discount[cur + 1];
                    low_sum += low_avg[(cur + 1) * paths + i];
                }
                low_avg[cur * paths + (ptr[cur]) % paths] = low_est / paths;
                ptr[cur]++;
                s.pop();
            }
            prev = cur;
        }
        low_ans = low_avg[0] / omp_get_num_threads();
        high_ans = high_avg[0] / omp_get_num_threads();
    }
    std::cout << low_ans<< "\n" << high_ans << "\n";
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    std::cout << duration_ms.count() << "\n";
    return 0;
}