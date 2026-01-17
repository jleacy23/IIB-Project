import numpy as np
from fxpmath import Fxp
from iib_project.cordic import Cordic

def test_atan_table():
    cordic = Cordic(iterations=20, word_length=14)
    atan_table = cordic.atan_table
    for i in range(8):
        expected_value = np.arctan(2 ** -i)
        actual_value = float(atan_table[i])
        assert np.isclose(actual_value, expected_value, atol=1e-3), f"Atan table value mismatch at index {i} of {actual_value} vs {expected_value}"
    print("Atan table test passed.")

def test_rotation():
    cordic = Cordic(iterations=20, word_length=14)
    ref_theta = Fxp(np.pi / 6).like(cordic.acc_t)
    x_in = Fxp(1.0).like(cordic.acc_t)
    y_in = Fxp(0.0).like(cordic.acc_t)

    x_ref = np.cos(ref_theta)
    y_ref = np.sin(ref_theta)

    x_out, y_out = cordic.rotate(x_in, y_in, ref_theta)
    
    assert np.isclose(np.array(x_out), np.array(x_ref), atol=1e-2), f"X output mismatch: {float(x_out)} vs {x_ref}"
    assert np.isclose(np.array(y_out), np.array(y_ref), atol=1e-2), f"Y output mismatch: {float(y_out)} vs {y_ref}"
    print("Rotation test passed.")

def test_complex_rotation():
    cordic = Cordic(iterations=20, word_length=14)
    ref_theta = Fxp(2* np.pi / 3).like(cordic.acc_t)
    z_in = Fxp(1.0).like(cordic.complex_t)
    z_ref = z_in * np.exp(1j * ref_theta)

    z_out = cordic.complex_rotate(z_in, ref_theta)
    z_ref = np.array(z_ref)
    z_out = np.array(z_out)
    assert np.isclose(np.real(z_out), np.real(z_ref), atol=1e-2), f"Real part mismatch: {float(np.real(z_out))} vs {np.real(z_ref)}"
    assert np.isclose(np.imag(z_out), np.imag(z_ref), atol=1e-2), f"Imaginary part mismatch: {float(np.imag(z_out))} vs {np.imag(z_ref)}"
    print("Complex rotation test passed.")

def test_complex_array_rotation():
    cordic = Cordic(iterations=20, word_length=14)
    ref_theta = Fxp(np.array([np.pi / 4, -np.pi / 4, np.pi / 1.5, np.pi / 3])).like(cordic.acc_t)
    z_in = Fxp(np.array([1.0, 0.0, -1.0, 1.0 + 1.0j])).like(cordic.complex_t)
    z_ref = z_in * np.exp(1j * ref_theta)

    z_out = cordic.complex_rotate(z_in, ref_theta)
    z_ref = np.array(z_ref)
    z_out = np.array(z_out)
    for i in range(len(z_in)):
        assert np.isclose(np.real(z_out[i]), np.real(z_ref[i]), atol=1e-2), f"Real part mismatch at index {i}: {float(np.real(z_out[i]))} vs {np.real(z_ref[i])}"
        assert np.isclose(np.imag(z_out[i]), np.imag(z_ref[i]), atol=1e-2), f"Imaginary part mismatch at index {i}: {float(np.imag(z_out[i]))} vs {np.imag(z_ref[i])}"
    print("Complex array rotation test passed.")

def test_negative_rotation():
    cordic = Cordic(iterations=20, word_length=14)
    ref_theta = Fxp(-np.pi / 4).like(cordic.acc_t)
    x_in = Fxp(1.0).like(cordic.acc_t)
    y_in = Fxp(0.0).like(cordic.acc_t)

    x_ref = np.cos(ref_theta)
    y_ref = np.sin(ref_theta)

    x_out, y_out = cordic.rotate(x_in, y_in, ref_theta)
    
    assert np.isclose(np.array(x_out), np.array(x_ref), atol=1e-2), f"X output mismatch: {float(x_out)} vs {x_ref}"
    assert np.isclose(np.array(y_out), np.array(y_ref), atol=1e-2), f"Y output mismatch: {float(y_out)} vs {y_ref}"
    print("Negative rotation test passed.")

def test_vectoring():
    cordic = Cordic(iterations=20, word_length=14)
    x_in = Fxp(-0.7071).like(cordic.acc_t)
    y_in = Fxp(-0.7071).like(cordic.acc_t)

    r_ref = np.sqrt(float(x_in)**2 + float(y_in)**2)
    theta_ref = np.arctan2(float(y_in), float(x_in))

    r_out, theta_out = cordic.vectoring(x_in, y_in)

    assert np.isclose(np.array(r_out), np.array(r_ref), atol=1e-2), f"Radius output mismatch: {float(r_out)} vs {r_ref}"
    assert np.isclose(np.array(theta_out), np.array(theta_ref), atol=1e-2), f"Theta output mismatch: {float(theta_out)} vs {theta_ref}"
    print("Vectoring test passed.")

def test_complex_phase():
    cordic = Cordic(iterations=20, word_length=14)
    z_in = Fxp(1 + 1j).like(cordic.complex_t)

    r_ref = np.abs(z_in)
    theta_ref = np.angle(z_in)
    
    theta_out = cordic.complex_phase(z_in)
    assert np.isclose(np.array(theta_out), np.array(theta_ref), atol=1e-2), f"Theta output mismatch: {float(theta_out)} vs {theta_ref}"
    print("Complex phase test passed.")

def test_complex_phase_array():
    cordic = Cordic(iterations=20, word_length=14)
    z_in = Fxp(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])/np.sqrt(2)).like(cordic.complex_t)

    theta_ref = np.angle(z_in)
    
    theta_out = cordic.complex_phase(z_in)
    assert np.allclose(np.array(theta_out), np.array(theta_ref), atol=1e-2), f"Theta output mismatch: {theta_out} vs {theta_ref}"
    print("Complex phase array test passed.")


if __name__ == "__main__":
    test_atan_table()
    test_rotation()
    test_complex_rotation()
    test_complex_array_rotation()
    test_negative_rotation()
    test_vectoring()
    test_complex_phase()
    test_complex_phase_array()
