// Package c64 holds tiny shared helpers for complex64 math that Go's
// standard library only provides for complex128 (via math/cmplx).
package c64

import "math"

// Conj returns the complex conjugate of a complex64.
func Conj(z complex64) complex64 {
	return complex(real(z), -imag(z))
}

// Norm2 returns |z|^2 (the squared magnitude).
func Norm2(z complex64) float32 {
	return real(z)*real(z) + imag(z)*imag(z)
}

// Arg returns the phase angle of z in (-pi, pi].
func Arg(z complex64) float32 {
	return float32(math.Atan2(float64(imag(z)), float64(real(z))))
}
