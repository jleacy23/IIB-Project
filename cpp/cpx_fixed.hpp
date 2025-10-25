#ifndef CPX_FIXED_HPP
#define CPX_FIXED_HPP

#include <ap_fixed.h>

// Templated complex number struct for fixed-point DSP/HLS
template<int DATA_WL, int DATA_IWL>
struct cpx {
    typedef ap_fixed<DATA_WL, DATA_IWL, AP_RND, AP_SAT> fx_t;

    fx_t re;
    fx_t im;

    // Constructors
    cpx() : re(0), im(0) {}
    cpx(fx_t r, fx_t i = 0) : re(r), im(i) {}
};

template<int WL, int IWL>
inline cpx<WL, IWL> operator+(const cpx<WL, IWL>& a, const cpx<WL, IWL>& b) {
    return cpx<WL, IWL>(a.re + b.re, a.im + b.im);
}

template<int WL, int IWL>
inline cpx<WL, IWL> operator-(const cpx<WL, IWL>& a, const cpx<WL, IWL>& b) {
    return cpx<WL, IWL>(a.re - b.re, a.im - b.im);
}

template<int WL, int IWL>
inline cpx<WL, IWL> conj(const cpx<WL, IWL>& a) {
    return cpx<WL, IWL>(a.re, -a.im);
}

template<int WL, int IWL>
inline cpx<WL, IWL> operator*(const cpx<WL, IWL>& a, const cpx<WL, IWL>& b) {
    // Wider accumulator helps prevent overflow
    typedef ap_fixed<2*WL, 2*IWL, AP_RND, AP_SAT> acc_t;
    acc_t ar = a.re;
    acc_t ai = a.im;
    acc_t br = b.re;
    acc_t bi = b.im;
    acc_t rr = ar * br - ai * bi;
    acc_t ii = ar * bi + ai * br;
    return cpx<WL, IWL>((typename cpx<WL,IWL>::fx_t)rr,
                        (typename cpx<WL,IWL>::fx_t)ii);
}

#endif // CPX_FIXED_HPP
