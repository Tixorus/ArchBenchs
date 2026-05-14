#ifndef STOCK_CPP
#define STOCK_CPP

#ifndef OPENRAND_DEVICE
#define OPENRAND_DEVICE 
#endif

#include <cmath>
#include "OpenRAND/include/openrand/philox.h"

#ifdef USE_MKL
#define CUSTOM_EXP(x) std::exp(x) 
#define CUSTOM_COS(x) std::cos(x)
#define CUSTOM_SIN(x) std::sin(x)
#define CUSTOM_LOG(x) std::log(x)
#else 

#define CUSTOM_EXP(x) std::exp(x) 
#define CUSTOM_COS(x) std::cos(x)
#define CUSTOM_SIN(x) std::sin(x)
#define CUSTOM_LOG(x) std::log(x)

// TODO: add the vector risc math funcs
template <typename RNG>
inline void generate_uniform_pair(RNG& rng, double& u1, double& u2)
{
    openrand::uint4 vals = rng.draw_int4();

    uint64_t r1 = (((uint64_t)vals.x) << 32) | ((uint64_t)vals.y);
    uint64_t r2 = (((uint64_t)vals.z) << 32) | ((uint64_t)vals.w);
    u1 = (r1 >> 11) * (1.0 / ((double)(1ull << 53)));
    u2 = (r2 >> 11) * (1.0 / ((double)(1ull << 53)));
    if (u1 <= 0.0) u1 = 1e-15;
}
#endif

struct Stock
{
    double S0;
    double volatility;
    double dividends = 0; //continuous
};

struct RetVal
{
    int time;
    double high_est;
    double low_est;
    double ans;
};

struct Option_EU
{
    double exercise_time;
    double strike_price;
    unsigned int stock_count;
    Stock* stocks;
    double* correlation;
};

#endif // !STOCK_CPP