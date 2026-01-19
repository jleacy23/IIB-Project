import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fxpmath import Fxp
from iib_project.carrier_recovery import Carrier_Recovery
from iib_project.modulator import Modulator
from iib_project.demodulator import Demodulator
from iib_project.channel import Channel
from iib_project.plotting import plot_constellation

def test_update_step_ref():
    symbol_rate = 32  # GHz
    sps = 2
    DW_acc = 16
    pilot_interval = 2**5
    N_pols = 2
    M = 4
    D = 16  # ps/(nm*km)
    L = 80  # km
    wavelength = 1550  # nm
    total_linewidth = 1e6  # Hz
    snr_db = 20  # dB
    symbol_energy = 1.0
    N = 5
    num_symbols = 2**18
    cordic_its = 16
    num_runs = 20



    cr = Carrier_Recovery(symbol_rate, sps, DW_acc, pilot_interval)
    mod = Modulator(M, N_pols)
    demod = Demodulator(M, N_pols)
    chan = Channel(SNR=20, sps=sps, symbol_rate=symbol_rate, D=D, L=L, wavelength=wavelength, total_linewidth=total_linewidth)

    
    x = mod.modulate(num_symbols)
    print(x.shape)
    pilots = np.angle(x[:, ::pilot_interval])
    print(pilots.shape)
    results = {}
    for _ in tqdm(range(num_runs)):
        x_noisy = chan.add_phase_noise(x)
        x_noisy = chan.add_AWGN(x_noisy)

        for step in [1,2,4,8,16,32,64,128, 256, 512, 1024]:
            y_ref = cr.viterbi_viterbi_ref(x_noisy, N, total_linewidth, snr_db, symbol_energy, step)
            y_dec = demod.decide(y_ref)

            ser = 1.0
            for v in range(4):
                for h in range(4):
                    for conj in [(False, False), (True, False), (False, True), (True, True)]:
                        y_rot = y_dec.copy()
                        y_rot[0] = y_rot[0] * (1j)**v
                        y_rot[1] = y_rot[1] * (1j)**h
                        if conj[0]:
                            y_rot[0] = np.conj(y_rot[0])
                        if conj[1]:
                            y_rot[1] = np.conj(y_rot[1])
                        ser_curr = np.sum(x != y_rot) / y_rot.size
                        if ser_curr < ser:
                            ser = ser_curr
            # add to average
            if step in results:
                results[str(step)] += ser / num_runs
            else:
                results[str(step)] = ser / num_runs

    print("SER results:", results)

    #convert to percentages
    for step in results:
        results[step] *= 100

    # Plot SER vs step size on bar chart
    plt.figure()
    steps = list(results.keys())
    sers = [results[step] for step in steps]
    plt.bar(steps, sers, width=0.5)
    plt.xlabel('Step Size')
    plt.ylabel('Symbol Error Rate (SER) [%]')
    plt.title(f'SER vs Number of Symbols per Viterbi-Viterbi Phase Estimate - Floating Point, Test Parameters: SNR={snr_db}dB, Linewidth={total_linewidth/1e3}kHz, Symbols={num_symbols}, Runs={num_runs}')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
def test_update_step_fxp():
    """ Perform same test as ref but with fxp implementation """
    symbol_rate = 32  # GHz
    sps = 2
    DW_acc = 16
    pilot_interval = 2**5
    N_pols = 2
    M = 4
    D = 16  # ps/(nm*km)
    L = 80  # km
    wavelength = 1550  # nm
    total_linewidth = 1e6  # Hz
    snr_db = 20  # dB
    symbol_energy = 1.0
    N = 5
    num_symbols = 2**12
    cordic_its = 16
    num_runs = 20



    cr = Carrier_Recovery(symbol_rate, sps, DW_acc, pilot_interval)
    mod = Modulator(M, N_pols)
    demod = Demodulator(M, N_pols)
    chan = Channel(SNR=20, sps=sps, symbol_rate=symbol_rate, D=D, L=L, wavelength=wavelength, total_linewidth=total_linewidth)

    
    x = mod.modulate(num_symbols)
    pilots = np.angle(x[:, ::pilot_interval])
    print(pilots.shape)
    results = {}
    for _ in tqdm(range(num_runs)):
        x_noisy = chan.add_phase_noise(x)
        x_noisy = chan.add_AWGN(x_noisy)
        scaling = np.percentile(np.abs(x_noisy), 95)
        x_noisy = x_noisy / scaling  # Normalize to avoid overflow in fxp
        x_fxp = Fxp(x_noisy).like(cr.acc_t)

        for step in [1,2,4,8,16, 32, 64, 128, 256, 512, 1024]:
            y_fxp = cr.viterbi_viterbi_fxp(x_fxp, N, total_linewidth, snr_db, symbol_energy/scaling**2, cordic_its, pilots, step)
            y_fxp_np = np.array(y_fxp)
            y_dec = demod.decide(y_fxp_np)

            ser = 1.0
            for v in range(4):
                for h in range(4):
                    for conj in [(False, False), (True, False), (False, True), (True, True)]:
                        y_rot = y_dec.copy()
                        y_rot[0] = y_rot[0] * (1j)**v
                        y_rot[1] = y_rot[1] * (1j)**h
                        if conj[0]:
                            y_rot[0] = np.conj(y_rot[0])
                        if conj[1]:
                            y_rot[1] = np.conj(y_rot[1])
                        ser_curr = np.sum(x != y_rot) / y_rot.size
                        if ser_curr < ser:
                            ser = ser_curr
            # add to average
            if step in results:
                results[str(step)] += ser / num_runs
            else:
                results[str(step)] = ser / num_runs
    print("SER results (FXP):", results)

    #convert to percentages
    for step in results:
        results[step] *= 100

    # Plot SER vs step size on bar chart
    plt.figure()
    steps = list(results.keys())
    sers = [results[step] for step in steps]
    plt.bar(steps, sers, width=0.5)
    plt.xlabel('Step Size')
    plt.ylabel('Symbol Error Rate (SER) [%]')
    plt.title(f'SER vs Number of Symbols per Viterbi-Viterbi Phase Estimate - Fixed Point')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    #test_update_step_ref()
    test_update_step_fxp()


        







