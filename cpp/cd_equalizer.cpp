#include <cmath>
#include <cstring>
#include <ap_int.h>
#include <ap_fixed.h>
#include "constants.hpp"
#include "cpx.hpp"


template<int N_FFT,
         double D,
         double L,
         double SYMBOL_RATE,
         double SPS,
         double CARRIER_WAVELENGTH,
         int DATA_WL = 8,
         int DATA_IWL = 2,
         bool USE_VENDOR_FFT = false
         >
class CD_Equalizer {

    static_assert((N_FFT & (N_FFT - 1)) == 0, "N_FFT must be a power of 2");

    public:
        //fixed point type
        typedef ap_fixed<DATA_WL, DATA_IWL, AP_RND, AP_SAT> fx_t; 
        // fixed point complex type
        typedef cpx<DATA_WL, DATA_IWL> cpx_t;
        // Impulse response length
        static const int N_CD = static_cast<int>(ceil(6.67 / (2 * PI * SPEED_OF_LIGHT) * CARRIER_WAVELENGTH^2 * D * L * SYMBOL_RATE^2 * OVERSAMPLING));
        // Nyquist frequency
        double NYQ_FREQ = SPS * SYMBOL_RATE / 2;

        static void equalize(const fx_t prev[N_FFT], const fx_t curr[N_FFT], const fx_t out[N_FFT]) {
            // Overlap and save method for CD equalization

            cpx_t H[N_FFT];
            get_H(H);

            cpx_t overlapped[N_FFT];
            overlapped[0..N_CD-1] = prev[ N_FFT - N_CD .. N_FFT -1 ];
            overlapped[N_CD..N_FFT-1] = curr[0..N_FFT - N_CD -1];

            cpx_t overlapped_fft[N_FFT];
            fft(overlapped, overlapped_fft, false);

            cpx_t equalized_fft[N_FFT];
            for (int i = 0; i < N_FFT; i++) {
                equalized_fft[i] = overlapped_fft[i] * H[i];
            }

            fft(equalized_fft, out, true);
        }

    private:
        static void fft(const cpx in[N_FFT], cpx out[N_FFT], bool inverse) {
            // Test FFT for fixed point arithmetic, should not be used by HLS tool

            // Copy input
            for (int i = 0; i < N_FFT; ++i) {
                out[i] = in[i];
            }

            // Bit-reversal reorder
            int log2N_FFT = 0;
            { int t = N_FFT; while (t >>= 1) ++log2N_FFT; }
            for (int i = 0; i < N_FFT; ++i) {
                int rev = bit_reverse(i, log2N_FFT);
                if (i < rev) {
                    cpx tmp = out[i];
                    out[i] = out[rev];
                    out[rev] = tmp;
                }
            }

            // Cooley-Tukey FFT
            for (int s = 1; s <= log2N_FFT; ++s) {
                int m = 1 << s;
                int m2 = m >> 1;
                // compute twiddle factors
                for (int k = 0; k < m2; ++k) {
                    double angle = (inverse ? 2.0 : -2.0) * N_FFT_PI * k / m;
                    double cosA = std::cos(angle);
                    double sinA = std::sin(angle);
                    cpx W;
                    W.re = static_cast<fx_t>(cosA);
                    W.im = static_cast<fx_t>(sinA);
                    for (int j = k; j < N_FFT; j += m) {
                        int t = j + m2;
                        cpx u = out[j];
                        cpx v = out[t] * W;
                        out[j] = u + v; 
                        out[t] = u - v; 
                    }
                }
            }

            // normalize IFFT 
            if (inverse) {
                for (int i = 0; i < N_FFT; ++i) {
                    out[i].re = out[i].re /static_cast<double>(N_FFT);
                    out[i].im = out[i].im / static_cast<double>(N_FFT);
                }
            }
        }

        static inline int bit_reverse(int x, int bits) {
            int y = 0;
            for (int i = 0; i < bits; ++i) {
                y = (y << 1) | (x & 1);
                x >>= 1;
            }
            return y;
        }

        static void get_H(cpx_t H[N_FFT]) {
            // Get frequency response of equalizer
            for (int i = 0; i < N_FFT; i++) {
                int f_idx = i - N_FFT i / 2;
                double f = f_idx * 2 * NYQ_FREQ / N_FFT;
                double phi = - PI * CARRIER_WAVELENGTH^2 * D * L * f^2 / SPEED_OF_LIGHT;
                double cos_phi = std::cos(phi);
                double sin_phi = std::sin(phi);
                H[i].re = static_cast<fx_t>(cos_phi);
                H[i].im = static_cast<fx_t>(sin_phi);
            }
        }
};

        





