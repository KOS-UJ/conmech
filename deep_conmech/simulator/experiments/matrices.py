
@njit
def denominator(x_i, x_j1, x_j2):
    return (
        x_i[1] * x_j1[0]
        + x_j1[1] * x_j2[0]
        + x_i[0] * x_j2[1]
        - x_i[1] * x_j2[0]
        - x_j2[1] * x_j1[0]
        - x_i[0] * x_j1[1]
    )


X:
{{(j2 k1 - j1 k2 - j2 l1 + k2 l1 + j1 l2 - k1 l2) /
(I2 j1 k0 - I1 j2 k0 - I2 j0 k1 + I0 j2 k1 + I1 j0 k2 - I0 j1 k2 - I2 j1 l0 + I1 j2 l0 +
I2 k1 l0 - j2 k1 l0 - I1 k2 l0 + j1 k2 l0 + I2 j0 l1 - I0 j2 l1 - I2 k0 l1 + j2 k0 l1 +
I0 k2 l1 - j0 k2 l1 - I1 j0 l2 + I0 j1 l2 + I1 k0 l2 - j1 k0 l2 - I0 k1 l2 + j0 k1 l2)}}

Y:
{{(j2 k0 - j0 k2 - j2 l0 + k2 l0 + j0 l2 - k0 l2) /
(-I2 j1 k0 + I1 j2 k0 + I2 j0 k1 - I0 j2 k1 - I1 j0 k2 + I0 j1 k2 + I2 j1 l0 - I1 j2 l0 -
I2 k1 l0 + j2 k1 l0 + I1 k2 l0 - j1 k2 l0 - I2 j0 l1 + I0 j2 l1 + I2 k0 l1 - j2 k0 l1 -
I0 k2 l1 + j0 k2 l1 + I1 j0 l2 - I0 j1 l2 - I1 k0 l2 + j1 k0 l2 + I0 k1 l2 - j0 k1 l2)}}


Z:
{{(j1 k0 - j0 k1 - j1 l0 + k1 l0 + j0 l1 - k0 l1) /
(I2 j1 k0 - I1 j2 k0 - I2 j0 k1 + I0 j2 k1 + I1 j0 k2 - I0 j1 k2 - I2 j1 l0 + I1 j2 l0 +
I2 k1 l0 - j2 k1 l0 - I1 k2 l0 + j1 k2 l0 + I2 j0 l1 - I0 j2 l1 - I2 k0 l1 + j2 k0 l1 +
I0 k2 l1 - j0 k2 l1 - I1 j0 l2 + I0 j1 l2 + I1 k0 l2 - j1 k0 l2 - I0 k1 l2 + j0 k1 l2)}}
