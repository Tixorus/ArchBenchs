#include <iostream>
#include <chrono>
#include "math.h"
#include "mkl.h"
#include <omp.h>
#include <cstdio>

//https://people.sc.fsu.edu/~jburkardt/cpp_src/toms462/toms462.html
#include "toms462.hpp"

#include "analytical.hpp"

double calc_real_price(double time, double strike_price, double interest_rate, Stock stock)
{
    double d1 = (log(stock.S0 / strike_price) + ((interest_rate - stock.dividends + stock.volatility * stock.volatility / 2) * time)) / (stock.volatility * sqrt(time));
    double d2 = d1 - stock.volatility * sqrt(time);
    double cdfd1 = 0;
    double cdfd2 = 0;
    vdCdfNorm(1, &d1, &cdfd1);
    vdCdfNorm(1, &d2, &cdfd2);
    return stock.S0 * cdfd1 - strike_price * exp((-interest_rate + stock.dividends) * time) * cdfd2;
}

//https://pure.uva.nl/ws/files/23194494/Thesis.pdf pg 27
//not 100% analytical due to bivariate cdf, but should be very accurate
double calc_real_price_two_no_div(double time, double strike_price, double interest_rate, double rho, Stock stock1, Stock stock2)
{
    double S1_adj = stock1.S0;
    double S2_adj = stock2.S0;

    double cA = (log(S1_adj / strike_price) + (interest_rate + stock1.volatility * stock1.volatility / 2) * time) / (stock1.volatility * sqrt(time));
    double cB = (log(S2_adj / strike_price) + (interest_rate + stock2.volatility * stock2.volatility / 2) * time) / (stock2.volatility * sqrt(time));
    double sigma = sqrt(stock2.volatility * stock2.volatility - 2 * rho * stock1.volatility * stock2.volatility + stock1.volatility * stock1.volatility);
    double rhoA = (stock1.volatility - rho * stock2.volatility) / sigma;
    double rhoB = (stock2.volatility - rho * stock1.volatility) / sigma;
    double cHatA = (log(S1_adj / S2_adj) + sigma * sigma * time / 2) / (sigma * sqrt(time));
    double cHatB = (log(S2_adj / S1_adj) + sigma * sigma * time / 2) / (sigma * sqrt(time));

    return S1_adj * bivnor(-cHatA, -cA, rhoA) + S2_adj * bivnor(-cHatB, -cB, rhoB) - strike_price * exp(-interest_rate * time) * (1 - bivnor(cA - stock1.volatility * sqrt(time), cB - stock2.volatility * sqrt(time), rho));
}