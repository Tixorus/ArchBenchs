#ifndef STOCK_CPP
#define STOCK_CPP

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

#endif // !STOCK_CPP