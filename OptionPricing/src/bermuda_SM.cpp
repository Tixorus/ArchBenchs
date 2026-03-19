#define _USE_MATH_DEFINES

#include "bermuda_SM.hpp"
#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>
#include <stack>


RetVal monte_carlo_bermuda_SM(int generator, long long b, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates)
{
    int buffer = 100 / stock_count;
    double* nodes = new double[stock_count * b * (exercise_num+1)];
    double* nodes_low = new double[stock_count * b * (exercise_num + 1)];
    double* high_est = new double[b * (exercise_num + 1)];
    double* low_est = new double[b];
    double* denom = new double[b* (exercise_num+1)];
    double* vol_sq_dt = new double[stock_count* (exercise_num + 1)];
    double* sqrt_pi_vol = new double[stock_count* (exercise_num + 1)];
    
    

    double drift[50], timesq[100], dt[100];
    for (int j = 0; j < stock_count; j++)
    {
        drift[j] = (interest_rate - stocks[j].dividends - 0.5 * stocks[j].volatility * stocks[j].volatility);
        for (int k = 0; k < b; k++)
        {
            nodes[k * stock_count + j] = stocks[j].S0;
            nodes_low[k * stock_count + j] = stocks[j].S0;
        }
    }
    timesq[0] = sqrt(exercise_dates[0]);
    dt[0] = exercise_dates[0];
    for (int j = 1; j < exercise_num; j++)
    {
        timesq[j] = sqrt(exercise_dates[j]- exercise_dates[j-1]);
        dt[j] = exercise_dates[j] - exercise_dates[j - 1];
    }
    for (int j = 0; j < b; j++)
    {
        low_est[j] = -1;
    }

    for (int cur_t = 0; cur_t < exercise_num; cur_t++)
    {
        for (int st = 0; st < stock_count; st++)
        {
            double vol = stocks[st].volatility;
            vol_sq_dt[(cur_t + 1) * stock_count + st] = 2.0 * vol * vol * dt[cur_t];
            sqrt_pi_vol[(cur_t + 1) * stock_count + st] = 1.0 / sqrt(M_PI * vol_sq_dt[(cur_t + 1) * stock_count + st]);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    

#pragma omp parallel
    {
        VSLStreamStatePtr stream;
        long long thread_id = omp_get_thread_num();
        double* gauss = new double[buffer*stock_count];
        if (generator == 0)
        {
            vslNewStream(&stream, VSL_BRNG_MCG59, 1234567);
            vslSkipAheadStream(stream, 2*thread_id * (b * stock_count * exercise_num / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 1)
        {
            vslNewStream(&stream, VSL_BRNG_SOBOL, stock_count);
            vslSkipAheadStream(stream, 2 * thread_id * (b * stock_count * exercise_num / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 2)
        {
            vslNewStream(&stream, VSL_BRNG_NIEDERR, stock_count);
            vslSkipAheadStream(stream, 2 * thread_id * (b * stock_count * exercise_num / omp_get_num_threads() + 1) + 1);
        }
        else if (generator == 3)
        {
            vslNewStream(&stream, VSL_BRNG_MT19937, 1234567);
            vslSkipAheadStream(stream, 2 * thread_id * (b * stock_count * exercise_num / omp_get_num_threads() + 1) + 1);
        }
        //mesh generation

        for (int cur_t = 0; cur_t < exercise_num; cur_t++)
        {
#pragma omp for schedule(static)
            for (int block = 0; block < b / buffer; block++)
            {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
                for (int i = 0; i < buffer; i++)
                {
                    for (int j = 0; j < stock_count; j++)
                    {
                        double d = stocks[j].volatility * timesq[cur_t] * gauss[j + i * stock_count];
                        double e = exp(drift[j]*dt[cur_t] + d);
                        nodes[(cur_t+1) * b * stock_count + (block * buffer + i) * stock_count + j] = nodes[cur_t*b*stock_count+(block*buffer+i)*stock_count+j] * e;
                    }
                }
            }
        }
        //high estimate
        double mx = 0;
#pragma omp for schedule(static)
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < stock_count; j++) mx = std::max(mx, nodes[exercise_num * b * stock_count + i * stock_count + j] - strike_price);
            high_est[exercise_num * b + i] = mx;
            mx = 0;
        }
        for (int cur_t = exercise_num-1; cur_t >= 0; cur_t--)
        {
#pragma omp for schedule(static)
            for (int next_node = 0; next_node < b; next_node++)
            {
                denom[(cur_t+1)*b + next_node] = 0;
                for (int cur_node = 0; cur_node < b; cur_node++)
                {
                    double cur_TD = 1;
                    for (int st = 0; st < stock_count; st++)
                    {
                        double alpha = log(nodes[(cur_t + 1) * b * stock_count + (next_node * stock_count) + st] / nodes[(cur_t)*b * stock_count + (cur_node * stock_count) + st]) - drift[st] * dt[cur_t];
                        cur_TD *= exp(-(alpha * alpha) / (vol_sq_dt[(cur_t + 1) * stock_count + st])) * sqrt_pi_vol[(cur_t + 1) * stock_count + st];
                    }
                    denom[(cur_t + 1) * b + next_node] += cur_TD;
                }
                denom[(cur_t + 1) * b + next_node]/=b;
            }
#pragma omp for schedule(static)
            for (int cur_node = 0; cur_node < b; cur_node++)
            {
                high_est[cur_t * b + cur_node] = 0;
                double ex_price = 0;
                for (int st = 0; st < stock_count; st++)
                {
                    ex_price = std::max(ex_price, nodes[cur_t * b * stock_count + cur_node * stock_count + st]-strike_price);
                }
                for (int next_node = 0; next_node < b; next_node++)
                {
                    double cur_TD = 1;
                    for (int st = 0; st < stock_count; st++)
                    {
                        double alpha = log(nodes[(cur_t + 1) * b * stock_count + (next_node * stock_count) + st] / nodes[(cur_t)*b * stock_count + (cur_node * stock_count) + st]) - drift[st] * dt[cur_t];
                        cur_TD *= exp(-(alpha * alpha) / (vol_sq_dt[(cur_t + 1) * stock_count + st])) * sqrt_pi_vol[(cur_t + 1) * stock_count + st];
                    }
                    high_est[cur_t * b + cur_node] += (high_est[(cur_t + 1) * b + next_node] * cur_TD) / denom[(cur_t + 1) * b + next_node];
                }
                high_est[cur_t * b + cur_node] /= b;
                high_est[cur_t * b + cur_node] *= exp(-interest_rate * dt[cur_t]);
                high_est[cur_t * b + cur_node] = std::max(high_est[cur_t * b + cur_node], ex_price);
            }
        }
        //low estimate
        for (int cur_t = 0; cur_t < exercise_num; cur_t++)
        {
#pragma omp for schedule(static)
            for (int block = 0; block < b / buffer; block++)
            {
                vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, buffer * stock_count, gauss, 0.0, 1.0);
                for (int i = 0; i < buffer; i++)
                {
                    for (int j = 0; j < stock_count; j++)
                    {
                        double d = stocks[j].volatility * timesq[cur_t] * gauss[j + i * stock_count];
                        double e = exp(drift[j] * dt[cur_t] + d);
                        nodes_low[(cur_t + 1) * b * stock_count + (block * buffer + i) * stock_count + j] = nodes_low[cur_t * b * stock_count + (block * buffer + i) * stock_count + j] * e;
                    }
                }
            }
        }
#pragma omp for schedule(static)
        for (int cur_path = 0; cur_path < b; cur_path++)
        {
            for (int cur_t = 1; cur_t < exercise_num; cur_t++)
            {
                double ex_price = 0;
                for (int st = 0; st < stock_count; st++)
                {
                    ex_price = std::max(ex_price, nodes_low[cur_t * b * stock_count + cur_path * stock_count + st]-strike_price);
                }
                double next_est = 0;
                for (int next_node = 0; next_node < b; next_node++)
                {
                    double cur_TD = 1;
                    for (int st = 0; st < stock_count; st++)
                    {
                        double alpha = log(nodes[(cur_t + 1) * b * stock_count + (next_node * stock_count) + st] / nodes_low[(cur_t)*b * stock_count + (cur_path * stock_count) + st]) - drift[st] * dt[cur_t];
                        cur_TD *= exp(-(alpha * alpha) / (vol_sq_dt[(cur_t + 1) * stock_count + st])) * sqrt_pi_vol[(cur_t + 1) * stock_count + st];
                    }
                    next_est += (high_est[(cur_t + 1) * b + next_node] * cur_TD) / denom[(cur_t + 1) * b + next_node];
                }
                next_est /= b;
                next_est *= exp(-interest_rate * dt[cur_t]);
                if (ex_price >= next_est)
                {
                    low_est[cur_path] = ex_price * exp(-interest_rate * exercise_dates[cur_t-1]);
                    break;
                }
            }
            if (low_est[cur_path] < 0)
            {
                double mx = 0;
                for (int j = 0; j < stock_count; j++) mx = std::max(mx, nodes_low[exercise_num * b * stock_count + cur_path * stock_count + j] - strike_price);
                low_est[cur_path] = mx * exp(-interest_rate * exercise_dates[exercise_num - 1]);
            }
        }
        vslDeleteStream(&stream);
        delete[] gauss;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    double low_ans = 0;
    for (int i = 0; i < b; i++) low_ans += low_est[i];

    RetVal ret;
    ret.high_est = high_est[0];
    ret.low_est = low_ans / b;
    ret.ans = (ret.high_est + ret.low_est) / 2;
    ret.time = duration_ms.count();

    delete[] nodes;
    delete[] nodes_low;
    delete[] high_est;
    delete[] low_est;
    delete[] denom;
    delete[] vol_sq_dt;
    delete[] sqrt_pi_vol;

    
    return ret;
}