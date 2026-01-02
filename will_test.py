import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

rect_filter = np.array([1, 1, 1, 1, 1])
convolved = signal.convolve(rect_filter, rect_filter, mode='full')
plt.stem(convolved)
plt.title('Convolution of Rectangular Filter with Itself')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
