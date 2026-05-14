#ifndef EUROPEAN_NUMERICAL_RISC_HPP
#define EUROPEAN_NUMERICAL_RISC_HPP

#include "stock.hpp"

void monte_carlo_eu_option_array(int generator, long long paths_per_option, Option_EU* options, int num_options, RetVal* results, int option_type, double interest_rate, uint64_t seed);

RetVal monte_carlo_cont_div_linear(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed);
RetVal monte_carlo_no_div_linear(int generator, long long paths, Option_EU option, double interest_rate, uint64_t seed);

#endif // EUROPEAN_NUMERICAL_RISC_HPP