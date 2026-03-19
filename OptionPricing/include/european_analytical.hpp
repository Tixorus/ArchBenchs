#ifndef ANALYTICAL_CPP
#define ANALYTICAL_CPP
#include "stock.hpp"

double calc_real_price(double time, double strike_price, double interest_rate, Stock stock);

double calc_real_price_two_no_div(double time, double strike_price, double interest_rate, double rho, Stock stock1, Stock stock2);

#endif