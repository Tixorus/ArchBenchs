#ifndef BERMUDA_CPP
#define BERMUDA_CPP
#include "stock.hpp"

double monte_carlo_bermuda_RT_linear(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates);

double monte_carlo_bermuda_RT_parallel(int generator, long long paths, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates);

#endif