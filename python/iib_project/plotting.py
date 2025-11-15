import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_constellation(tx_data: np.ndarray, title="Transmitted Constellation"):
    """ Plots constellation diagram"""
    plt.figure(figsize=(6, 6))
    plt.scatter(tx_data.real, tx_data.imag, color='blue', s=1)
    plt.title(title)
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_rx_constellation(rx_data: np.ndarray, ber: float, snr: float, title="Received Constellation"):
    """ Plots received constellation and reports BER for a given SNR (dB)"""
    plt.figure(figsize=(6, 6))
    plt.scatter(rx_data.real, rx_data.imag, color='red', s=1)
    plt.title(f"{title}\nSNR: {snr} dB, BER: {ber:.2e}")
    plt.xlabel('In-Phase')
    plt.ylabel('Quadrature')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_ber_vs_snr(snr_vals: np.ndarray, ber_vals: np.ndarray, title="BER vs SNR"):
    """ Plots BER vs SNR (dB)"""
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_vals, ber_vals, marker='o', linestyle='-', color='green')
    plt.title(title)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(1e-6, 1)
    plt.xlim(min(snr_vals), max(snr_vals))
    plt.show()

def plot_time_domain(signal: np.ndarray, title="Time Domain Signal"):
    """ Plots In-Phase and Quadrature components of the time domain signal"""
    # 2 subplots: one for In-Phase and one for Quadrature
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(signal.real, color='blue')
    plt.title("In-Phase Component")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(signal.imag, color='orange')
    plt.title("Quadrature Component")
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
