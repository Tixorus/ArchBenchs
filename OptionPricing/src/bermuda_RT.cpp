#include "bermuda_RT.hpp"
#include <iostream>
#include <chrono>
#include "math.h"
#ifdef USE_MKL
#include "mkl.h"
#endif
#include <omp.h>
#include <cstdio>
#include <stack>



RetVal monte_carlo_bermuda_RT_linear(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    double prevtime = 0;
    double* drift = new double[(exercise_num + 1) * stock_count];
    double* discount = new double[exercise_num + 1];
    double* timesq = new double[exercise_num + 1];
    double* high_avg = new double[exercise_num + 1];
    int* ptr = new int[exercise_num + 1];
    double* low_avg = new double[paths * (exercise_num + 1)];

    high_avg[0] = 0;
    ptr[0] = 0;

    for (int i = 1; i <= exercise_num; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            drift[i * stock_count + j] = CUSTOM_EXP((interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * (exercise_dates[i - 1] - prevtime));
        }
        discount[i] = CUSTOM_EXP(-interest_rate * (exercise_dates[i - 1] - prevtime));
        timesq[i] = sqrt(exercise_dates[i - 1] - prevtime);
        prevtime = exercise_dates[i - 1];
        high_avg[i] = 0;
        ptr[i] = 0;
    }

    auto start = std::chrono::high_resolution_clock::now();

#ifdef USE_MKL
    VSLStreamStatePtr stream;
#endif
    uint32_t thread_id = omp_get_thread_num();
    double* gauss = new double[stock_count];

#ifdef USE_MKL
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
#endif

    std::stack<int> s;
    double* prices_ar = new double[stock_count * (exercise_num + 1)];
    int prev = 0;
    s.push(0);

#ifndef USE_MKL
    openrand::Philox rng(1234567, thread_id);
    bool has_spare = false;
    double spare_gauss = 0.0;
#endif

    for (int i = 0; i < stock_count; i++) prices_ar[i] = stocks[i].S0;
    for (int i = 0; i < paths; i++) s.push(1);
    while (!s.empty())
    {
        int cur = s.top();
        if (cur >= prev)
        {
#ifdef USE_MKL
            vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, stock_count, gauss, 0.0, 1.0);
#else
            for (int g_idx = 0; g_idx < stock_count; g_idx++)
            {
                if (has_spare)
                {
                    gauss[g_idx] = spare_gauss;
                    has_spare = false;
                }
                else
                {
                    double u1, u2;
                    generate_uniform_pair(rng, u1, u2);
                    double radius = sqrt(-2.0 * CUSTOM_LOG(u1));
                    double theta = 2.0 * 3.14159265358979323846 * u2;
                    gauss[g_idx] = radius * CUSTOM_COS(theta);
                    spare_gauss = radius * CUSTOM_SIN(theta);
                    has_spare = true;
                }
            }
#endif
            for (int i = 0; i < stock_count; i++) prices_ar[cur * stock_count + i] = prices_ar[(cur - 1) * stock_count + i] * drift[cur * stock_count + i] * CUSTOM_EXP(stocks[i].volatility * timesq[cur] * gauss[i]);

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
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    RetVal ret;
    ret.high_est = high_avg[0];
    ret.low_est = low_avg[0];
    ret.ans = (low_avg[0] + high_avg[0]) / 2;
    ret.time = duration_ms.count();

#ifdef USE_MKL
    vslDeleteStream(&stream);
#endif
    delete[] drift;
    delete[] discount;
    delete[] timesq;
    delete[] ptr;
    delete[] high_avg;
    delete[] low_avg;
    delete[] gauss;
    delete[] prices_ar;

    return ret;
}

RetVal monte_carlo_bermuda_RT_parallel(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    double prevtime = 0;
    double* drift = new double[(exercise_num + 1) * stock_count];
    double* discount = new double[exercise_num + 1];
    double* timesq = new double[exercise_num + 1];

    for (int i = 1; i <= exercise_num; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            drift[i * stock_count + j] = CUSTOM_EXP((interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * (exercise_dates[i - 1] - prevtime));
        }
        discount[i] = CUSTOM_EXP(-interest_rate * (exercise_dates[i - 1] - prevtime));
        timesq[i] = sqrt(exercise_dates[i - 1] - prevtime);
        prevtime = exercise_dates[i - 1];
    }

    auto start = std::chrono::high_resolution_clock::now();
    double low_ans = 0, high_ans = 0;

#pragma omp parallel reduction(+:low_ans, high_ans)
    {
#ifdef USE_MKL
        VSLStreamStatePtr stream;
#endif
        uint32_t thread_id = omp_get_thread_num();
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
#ifdef USE_MKL
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
#endif
        std::stack<int> s;
        double* prices_ar = new double[stock_count * (exercise_num + 1)];
        int prev = 0;
        s.push(0);

#ifndef USE_MKL
        openrand::Philox rng(1234567, thread_id);
        bool has_spare = false;
        double spare_gauss = 0.0;
#endif

        for (int i = 0; i < stock_count; i++) prices_ar[i] = stocks[i].S0;
        for (int i = 0; i < paths; i++) s.push(1);
        while (!s.empty())
        {
            int cur = s.top();
            if (cur >= prev)
            {
#ifdef USE_MKL
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, stock_count, gauss, 0.0, 1.0);
#else
                for (int g_idx = 0; g_idx < stock_count; g_idx++)
                {
                    if (has_spare)
                    {
                        gauss[g_idx] = spare_gauss;
                        has_spare = false;
                    }
                    else
                    {
                        double u1, u2;
                        generate_uniform_pair(rng,u1, u2);
                        double radius = sqrt(-2.0 * CUSTOM_LOG(u1));
                        double theta = 2.0 * 3.14159265358979323846 * u2;
                        gauss[g_idx] = radius * CUSTOM_COS(theta);
                        spare_gauss = radius * CUSTOM_SIN(theta);
                        has_spare = true;
                    }
                }
#endif
                for (int i = 0; i < stock_count; i++) prices_ar[cur * stock_count + i] = prices_ar[(cur - 1) * stock_count + i] * drift[cur * stock_count + i] * CUSTOM_EXP(stocks[i].volatility * timesq[cur] * gauss[i]);

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
#ifdef USE_MKL
        vslDeleteStream(&stream);
#endif
        delete[] ptr;
        delete[] high_avg;
        delete[] low_avg;
        delete[] gauss;
        delete[] prices_ar;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    RetVal ret;
    ret.high_est = high_ans;
    ret.low_est = low_ans;
    ret.ans = (low_ans + high_ans) / 2;
    ret.time = duration_ms.count();

    delete[] drift;
    delete[] discount;
    delete[] timesq;


    return ret;
}
//--------------------------------------------------------------------------------------------------------------------------


struct NodeResult
{
    double high_est;
    double low_est;
};

NodeResult process_subtree(
#ifdef USE_MKL
    VSLStreamStatePtr stream,
#else
    openrand::Philox& rng,
#endif
    int cur_depth,
    int max_depth,
    int paths,
    int stock_count,
    Stock* stocks,
    double* drift,
    double* discount,
    double* timesq,
    double strike_price,
    double* gauss,
    double* low_est,
    double* prices_ar)
{
    double exercise_price = 0;
    for (int i = 0; i < stock_count; i++)
    {
        exercise_price = std::max(exercise_price, prices_ar[(cur_depth - 1) * stock_count + i] - strike_price);
    }

    if (cur_depth == max_depth)
    {
        return { exercise_price, exercise_price };
    }

    double high_avg = 0;
    double low_avg = 0;
    double low_sum = 0;
    NodeResult res;

#ifdef USE_MKL
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, paths * stock_count, gauss + (paths * stock_count * cur_depth), 0.0, 1.0);
#else
    bool has_spare = false;
    double spare_gauss = 0.0;
    int total_gauss = paths * stock_count;
    double* current_gauss_ptr = gauss + (paths * stock_count * cur_depth);
    for (int g_idx = 0; g_idx < total_gauss; g_idx++)
    {
        if (has_spare)
        {
            current_gauss_ptr[g_idx] = spare_gauss;
            has_spare = false;
        }
        else
        {
            double u1, u2;
            generate_uniform_pair(rng, u1, u2);
            double radius = sqrt(-2.0 * CUSTOM_LOG(u1));
            double theta = 2.0 * 3.14159265358979323846 * u2;
            current_gauss_ptr[g_idx] = radius * CUSTOM_COS(theta);
            spare_gauss = radius * CUSTOM_SIN(theta);
            has_spare = true;
        }
    }
#endif

    for (int i = 0; i < paths; i++)
    {
        for (int j = 0; j < stock_count; j++) prices_ar[cur_depth * stock_count + j] = prices_ar[(cur_depth - 1) * stock_count + j] * drift[cur_depth * stock_count + j] * CUSTOM_EXP(stocks[j].volatility * timesq[cur_depth] * gauss[(paths * stock_count * cur_depth) + i * stock_count + j]);
        res = process_subtree(
#ifdef USE_MKL
            stream,
#else
            rng,
#endif
            cur_depth + 1, max_depth, paths, stock_count, stocks, drift, discount, timesq, strike_price, gauss, low_est, prices_ar);
        high_avg += res.high_est * discount[cur_depth] / paths; //note the lack of +1 here
        low_est[cur_depth * paths + i] = res.low_est; //and here
        low_sum += res.low_est;
    }
    high_avg = std::max(exercise_price, high_avg);

    for (int i = 0; i < paths; i++)
    {
        low_sum -= low_est[cur_depth * paths + i];
        if (exercise_price >= (low_sum / (paths - 1)) * discount[cur_depth]) low_avg += exercise_price;
        else low_avg += (low_est[cur_depth * paths + i]) * discount[cur_depth];
        low_sum += low_est[(cur_depth)*paths + i];
    }
    low_avg /= paths;
    return { high_avg, low_avg };
}

void gen_two_levels(int generator,
    long long paths,
    double strike_price,
    double interest_rate,
    unsigned int stock_count,
    Stock* stocks,
    int exercise_num,
    double* exercise_dates,
    double* prices_two_levels,
    double* drift,
    double* timesq)
{
    double* gauss_two_levels = new double[(paths + 1) * paths * stock_count];
#ifdef USE_MKL
    VSLStreamStatePtr stream;
    if (generator == 0)
    {
        vslNewStream(&stream, VSL_BRNG_MCG59, 1234567);
    }
    else if (generator == 1)
    {
        vslNewStream(&stream, VSL_BRNG_MT19937, 1234567);
    }
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, (paths + 1) * paths * stock_count, gauss_two_levels, 0.0, 1.0);
#else
    uint32_t thread_id = omp_get_thread_num();
    openrand::Philox rng(123456789, thread_id);
    bool has_spare = false;
    double spare_gauss = 0.0;
    long long total_gauss = (paths + 1) * paths * stock_count;
    for (long long g_idx = 0; g_idx < total_gauss; g_idx++)
    {
        if (has_spare)
        {
            gauss_two_levels[g_idx] = spare_gauss;
            has_spare = false;
        }
        else
        {
            double u1, u2;
            generate_uniform_pair(rng, u1, u2);
            double radius = sqrt(-2.0 * CUSTOM_LOG(u1));
            double theta = 2.0 * 3.14159265358979323846 * u2;
            gauss_two_levels[g_idx] = radius * CUSTOM_COS(theta);
            spare_gauss = radius * CUSTOM_SIN(theta);
            has_spare = true;
        }
    }
#endif

    for (int i = 0; i < paths; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            prices_two_levels[i * stock_count + j] = stocks[j].S0 * drift[j] * CUSTOM_EXP(stocks[j].volatility * timesq[0] * gauss_two_levels[i * stock_count + j]);
        }
    }
    for (int a = 1; a <= paths; a++)
    {
        for (int i = 0; i < paths; i++)
        {
            for (int j = 0; j < stock_count; j++)
            {
                prices_two_levels[a * paths * stock_count + i * stock_count + j] = prices_two_levels[(a - 1) * stock_count + j] * drift[stock_count + j] * CUSTOM_EXP(stocks[j].volatility * timesq[1] * gauss_two_levels[a * paths * stock_count + i * stock_count + j]);
            }
        }
    }
#ifdef USE_MKL
    vslDeleteStream(&stream);
#endif
    delete[] gauss_two_levels;
    return;
}

RetVal monte_carlo_bermuda_RT_recursive_parallel(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    double prevtime = 0;
    double* drift = new double[(exercise_num)*stock_count];
    double* discount = new double[exercise_num];
    double* timesq = new double[exercise_num];
    double* prices_two_levels = new double[(paths + 1) * paths * stock_count];

    for (int i = 0; i < exercise_num; i++)
    {
        for (int j = 0; j < stock_count; j++)
        {
            drift[i * stock_count + j] = CUSTOM_EXP((interest_rate - 0.5 * stocks[j].volatility * stocks[j].volatility) * (exercise_dates[i] - prevtime));
        }
        discount[i] = CUSTOM_EXP(-interest_rate * (exercise_dates[i] - prevtime));
        timesq[i] = sqrt(exercise_dates[i] - prevtime);
        prevtime = exercise_dates[i];
    }
    auto start = std::chrono::high_resolution_clock::now();

    gen_two_levels(generator, paths, strike_price, interest_rate, stock_count, stocks, exercise_num, exercise_dates, prices_two_levels, drift, timesq);

    long long points_num = 0;
    long long p_mult = paths;
    for (int i = 2; i < exercise_num; i++)
    {
        points_num += p_mult;
        p_mult *= paths;
    }

    double* lvl2_high = new double[paths * paths];
    double* lvl2_low = new double[paths * paths];
#pragma omp parallel
    {
        uint32_t thread_id = omp_get_thread_num();
#ifdef USE_MKL
        VSLStreamStatePtr stream;
#else
        openrand::Philox rng(1234567, thread_id);
#endif
        
        int threads_nums = omp_get_num_threads();

        double* gauss = new double[paths * stock_count * exercise_num];

        double* low_est = new double[paths * (exercise_num)];
#ifdef USE_MKL
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, 1234567);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, 1234567);
        }
        vslSkipAheadStream(stream, (paths + 1) * paths * stock_count + thread_id * (points_num) * (paths * paths * stock_count) / (threads_nums)+1);
#endif
        double* prices_ar = new double[stock_count * (exercise_num)];
        NodeResult res;
#pragma omp for schedule(static)
        for (int i = 0; i < paths * paths; i++)
        {
            for (int j = 0; j < stock_count; j++)
            {
                prices_ar[stock_count + j] = prices_two_levels[(paths + i) * stock_count + j];
                prices_ar[j] = prices_two_levels[(i / paths) * stock_count + j];
            }
            res = process_subtree(
#ifdef USE_MKL
                stream,
#else
                rng,
#endif
                2, exercise_num, paths, stock_count, stocks, drift, discount, timesq, strike_price, gauss, low_est, prices_ar);
            lvl2_high[i] = res.high_est;
            lvl2_low[i] = res.low_est;
        }
        delete[] gauss;
        delete[] low_est;
        delete[] prices_ar;
#ifdef USE_MKL
        vslDeleteStream(&stream);
#endif
    }
    double low_ans = 0;
    double high_ans = 0;
    for (int i = 0; i < paths; i++)
    {
        double exercise_price = 0;
        for (int j = 0; j < stock_count; j++)
        {
            exercise_price = std::max(exercise_price, prices_two_levels[i * stock_count + j] - strike_price);
        }
        double sum_low = 0;
        double low_avg = 0;
        double high_avg = 0;
        for (int j = 0; j < paths; j++) sum_low += lvl2_low[i * paths + j];
        for (int j = 0; j < paths; j++)
        {
            high_avg += lvl2_high[i * paths + j] * discount[1];
            sum_low -= lvl2_low[i * paths + j];
            if (exercise_price >= (sum_low / (paths - 1)) * discount[1]) low_avg += exercise_price;
            else low_avg += (lvl2_low[i * paths + j]) * discount[1];
            sum_low += lvl2_low[i * paths + j];
        }
        low_avg /= paths;
        high_avg /= paths;
        high_avg = std::max(exercise_price, high_avg);
        low_ans += low_avg * discount[0];
        high_ans += high_avg * discount[0];

    }
    low_ans /= paths;
    high_ans /= paths;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    RetVal ret;
    ret.high_est = high_ans;
    ret.low_est = low_ans;
    ret.ans = (low_ans + high_ans) / 2;
    ret.time = duration_ms.count();

    delete[] drift;
    delete[] discount;
    delete[] timesq;
    delete[] prices_two_levels;
    delete[] lvl2_high;
    delete[] lvl2_low;
    return ret;
}