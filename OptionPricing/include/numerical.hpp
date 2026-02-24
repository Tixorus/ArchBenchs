#ifndef NUMERICAL_CPP
#define NUMERICAL_CPP
#include "stock.hpp"

double monte_carlo_cont_div(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int seed);

double monte_carlo_no_div(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int seed);

double monte_carlo_multivariate(int generator, long long paths, double time, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, double* correlation, int seed);

#endif