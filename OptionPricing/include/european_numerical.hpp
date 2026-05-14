#ifndef NUMERICAL_CPP
#define NUMERICAL_CPP
#include "stock.hpp"

RetVal monte_carlo_cont_div(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed);

RetVal monte_carlo_no_div(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed);

RetVal monte_carlo_multivariate(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed);

#endif