import numpy as np
from iib_project.cordic import Cordic

def test_atan_table():
    cordic = Cordic(iterations=16, word_length=16)
    atan_table = cordic.atan_table
    for i in range(16):
        expected_value = np.arctan(2 ** -i)
        actual_value = float(atan_table[i])
        assert np.isclose(actual_value, expected_value, atol=1e-3), f"Atan table value mismatch at index {i} of {actual_value} vs {expected_value}"
    print("Atan table test passed.")

def test_rotation():
    cordic = Cordic(iterations=20, word_length=16)
    ref_theta = np.pi / 6
    x_in = 1.0
    y_in = 0.0

    x_ref = np.cos(ref_theta)
    y_ref = np.sin(ref_theta)

    x_out, y_out = cordic.rotate(x_in, y_in, ref_theta)
    
    assert np.isclose(float(x_out), x_ref, atol=1e-3), f"X output mismatch: {float(x_out)} vs {x_ref}"
    assert np.isclose(float(y_out), y_ref, atol=1e-3), f"Y output mismatch: {float(y_out)} vs {y_ref}"
    print("Rotation test passed.")

def test_complex_rotation():
    cordic = Cordic(iterations=20, word_length=16)
    ref_theta = np.pi / 4
    z_in = 1.0
    z_ref = z_in * np.exp(1j * ref_theta)

    z_out = cordic.complex_rotate(z_in, ref_theta)
    assert np.isclose(float(np.real(z_out)), np.real(z_ref), atol=1e-3), f"Real part mismatch: {float(np.real(z_out))} vs {np.real(z_ref)}"
    assert np.isclose(float(np.imag(z_out)), np.imag(z_ref), atol=1e-3), f"Imaginary part mismatch: {float(np.imag(z_out))} vs {np.imag(z_ref)}"
    print("Complex rotation test passed.")

def test_complex_array_rotation():
    cordic = Cordic(iterations=20, word_length=16)
    ref_theta = np.pi / 3
    z_in = np.array([1.0, 0.0, -1.0, 1.0 + 1.0j])
    z_ref = z_in * np.exp(1j * ref_theta)

    z_out = np.array([cordic.complex_rotate(z, ref_theta) for z in z_in])
    for i in range(len(z_in)):
        assert np.isclose(float(np.real(z_out[i])), np.real(z_ref[i]), atol=1e-3), f"Real part mismatch at index {i}: {float(np.real(z_out[i]))} vs {np.real(z_ref[i])}"
        assert np.isclose(float(np.imag(z_out[i])), np.imag(z_ref[i]), atol=1e-3), f"Imaginary part mismatch at index {i}: {float(np.imag(z_out[i]))} vs {np.imag(z_ref[i])}"
    print("Complex array rotation test passed.")


if __name__ == "__main__":
    test_atan_table()
    test_rotation()
    test_complex_rotation()
    test_complex_array_rotation()
