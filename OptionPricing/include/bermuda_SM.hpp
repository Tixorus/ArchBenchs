#ifndef BERMUDA_SM_CPP
#define BERMUDA_SM_CPP
#include "stock.hpp"

RetVal monte_carlo_bermuda_SM(int generator, long long b, double strike_price, double interest_rate, unsigned int stock_count, Stock* stocks, int exercise_num, double* exercise_dates);

#endif