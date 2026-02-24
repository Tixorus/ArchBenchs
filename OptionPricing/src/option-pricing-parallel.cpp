#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>

#include "analytical.hpp"
#include "numerical.hpp"
#include "bermuda.hpp"

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
    int stock_count = 20;
    
    srand(time(NULL));
    Stock Ar[50];
    long long pathsar[6] = { 50000, 500000, 5000000, 50000000, 200000000, 500000000 };
    //freopen("output.txt", "w", stdout);
    for (int i = 0; i < 10; i++)
    {
        Stock A;
        Stock B;
        A.S0 = 200;
        B.S0 = 200;
        A.volatility = 0.5;
        B.volatility = 0.3;
        A.dividends = 0.0;
        B.dividends = 0.0;
        Ar[2*i] = A;
        Ar[2 * i + 1] = B;
    }

    omp_set_num_threads(12);

    double ans = 0;
    double analytical = calc_real_price(2, 200, 0.7, Ar[0]);
    std::cout << analytical << "\n\n\n";
    double tim[] = {0.5, 1, 1.5, 2};
    omp_set_num_threads(6);
    monte_carlo_bermuda_RT_parallel(0,40, 200, 0.7, 1, Ar, 4, tim);

   /* double *mat = new double[stock_count*(stock_count+1)/2];
    generate_correlation_matrix(stock_count, mat); 
    int info = 0;
    dpptrf("L", (const int*)&stock_count, mat, &info);*/

    //for (int seeds = 0; seeds < 100; seeds++)
    //{
    //    int seed = ((double)rand() / RAND_MAX) * 2147483647;
    //    for (int p = 0; p < 6; p++)
    //    {
    //        //cout << std::fixed << seed << "\n";
    //        ans = monte_carlo_multivariate(1, pathsar[p], 0.5, 100, 0.03, stock_count, Ar, mat, seed);
    //        std::cout << std::scientific << ans - analytical << "\n";
    //    }
    //    std::cout << "\n";
    //}
    //cout << abs(ans / analytical - 1) << " - rel";
    /*Stock A;
    A.S0 = 200;
    A.volatility = 0.2;
    std::cout << monte_carlo(false, 1000000000, 0.5, 100, 0.03, 1, &A) - calc_real_price(0.5, 100, 0.03, A) << "\n";*/
    return 0;
}