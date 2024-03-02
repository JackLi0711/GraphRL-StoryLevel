'''
beam sections:
H x B x tw x tf
(1)  350x200x9x16
(2)  350x200x9x20
(3)  350x200x12x22
(4)  375x200x12x25
(5)  375x250x12x25
(6)  375x250x12x28
(7)  400x250x16x32
(8)  400x250x16x36
(9)  400x300x16x36

(10) B450x300x18x36_
(11) B450x300x18x40_
(12) B450x300x18x45_
(13) B500x350x20x40_
(14) B500x350x20x45_
(15) B500x350x20x50_

(10) 500x350x16x40
(11) 500x350x20x45
(12) 500x350x22x45
(13) 600x350x25x45
(14) 600x350x25x50
(15) 600x400x25x50

column sections:
Dx x Dy x t
(1)  350x350x20
(2)  350x350x22
(3)  350x350x25
(4)  375x375x28
(5)  375x375x32
(6)  375x375x36
(7)  400x400x40
(8)  400x400x45
(9)  400x400x50

(10) C450x450x46_
(11) C450x450x48_
(12) C450x450x50_
(13) C500x500x46_
(14) C500x500x48_
(15) C500x500x50_

(10) 500x500x40
(11) 500x500x45
(12) 500x500x50
(13) 600x600040
(14) 600x600x45
(15) 600x600x50
'''


# YIELDING_STRESS = 350 # 350 MPa = 350 x 10^3 kN/m^2, so My (kN x mm) = S (cm^3) x 350 (MPa) --> kN x mm
# convert Fy to kN, mm
YIELDING_STRESS = 350 * 1e+3 * 1e-6     # kN/mm^2
# so My then become --> My(kN x mm) = S (cm3) * 1e+3 * Fy (kN/mm^2) = kN x mm

# I-section shape factor = 1.12 ~ 1.14
# Rectangular shape factor = 1.5
BEAM_SHAPE_FACTOR = 1.12
COLUMN_SHAPE_FACTOR = 1.5

# I-beam effect shear area
# Asy = H * tw
# Asz = 5/6 * (2 * B * tf)
# Hollow rectangular shear area
# Asy = 2 * H * tw
# Asz = 2 * B * tf


# Various beam sections
# Original
B350x200x9x16 = {
     'name': '350x200x9x16',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 16,
     't_w(mm)': 9,
     'A(cm2)': 92.62,
     'J(cm4)': 61.836,
     'I_y(cm4)': 2135.265,
     'I_z(cm4)': 20274.4207,
     'S_y(cm3)': 213.526,
     'S_z(cm3)': 1158.538,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (16/10)),
     'My_z(kN-mm)': 1158.538 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1158.538 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 213.526 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1158.538 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [157, 227, 245]
}

B350x200x9x20 = {
     'name': '350x200x9x20',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 20,
     't_w(mm)': 9,
     'A(cm2)': 107.9,
     'J(cm4)': 110.729,
     'I_y(cm4)': 2668.549,
     'I_z(cm4)': 24040.991,
     'S_y(cm3)': 266.855,
     'S_z(cm3)': 1373.770,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (20/10)),
     'My_z(kN-mm)': 1373.770 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1373.770 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 266.855 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1373.770 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [139, 196, 240]
}

B350x200x12x22 = {
     'name': '350x200x12x22',
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 22,
     't_w(mm)': 12,
     'A(cm2)': 124.72,
     'J(cm4)': 155.109,
     'I_y(cm4)': 2937.739,
     'I_z(cm4)': 26569.234,
     'S_y(cm3)': 293.773,
     'S_z(cm3)': 1518.241,
     'Av_y(cm2)': (350/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (22/10)),
     'My_z(kN-mm)': 1518.241 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1518.241 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 293.773 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1518.241 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [122, 165, 235]
}

B375x200x12x25 = {
     'name': '375x200x12x25',
     'H(mm)': 375,
     'B(mm)': 200,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 139.00,
     'J(cm4)': 217.388,
     'I_y(cm4)': 3338.013,
     'I_z(cm4)': 34109.895,
     'S_y(cm3)': 333.801,
     'S_z(cm3)': 1819.194,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (25/10)),
     'My_z(kN-mm)': 1819.194 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1819.194 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 333.801 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1819.194 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [105, 134, 230]
}

B375x250x12x25 = {
     'name': '375x250x12x25',
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 164.00,
     'J(cm4)': 269.773,
     'I_y(cm4)': 6515.096,
     'I_z(cm4)': 41779.166,
     'S_y(cm3)': 521.207,
     'S_z(cm3)': 2228.222,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (25/10)),
     'My_z(kN-mm)': 2228.222 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2228.222 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 521.207 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2228.222 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [88, 104, 225]
}

B375x250x12x28 = {
     'name': '375x250x12x28',
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 28,
     't_w(mm)': 12,
     'A(cm2)': 178.28,
     'J(cm4)': 367.004,
     'I_y(cm4)': 7296.260,
     'I_z(cm4)': 45480.792,
     'S_y(cm3)': 583.700,
     'S_z(cm3)': 2425.700,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (28/10)),
     'My_z(kN-mm)': 2425.700 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2425.700 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 583.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2425.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [71, 73, 220]
}

B400x250x12x32 = {
     'name': '400x250x12x32',
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 32,
     't_w(mm)': 12,
     'A(cm2)': 200.32,
     'J(cm4)': 532.235,
     'I_y(cm4)': 8338.171,
     'I_z(cm4)': 58099.438,
     'S_y(cm3)': 667.053,
     'S_z(cm3)': 2904.971,
     'Av_y(cm2)': (400/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (32/10)),
     'My_z(kN-mm)': 2904.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2904.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 667.053 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2904.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 42, 215]
}

B400x250x16x36 = {
     'name': '400x250x16x36',
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 232.48,
     'J(cm4)': 774.870,
     'I_y(cm4)': 9386.195,
     'I_z(cm4)': 64522.606,
     'S_y(cm3)': 750.895,
     'S_z(cm3)': 3226.130,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (36/10)),
     'My_z(kN-mm)': 3226.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3226.130 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 750.895 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3226.130 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [37, 12, 211]
}

B400x300x16x36 = {
     'name': '400x300x16x36',
     'H(mm)': 400,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 268.48,
     'J(cm4)': 931.655,
     'I_y(cm4)': 16211.195,
     'I_z(cm4)': 76486.126,
     'S_y(cm3)': 1080.746,
     'S_z(cm3)': 3824.306,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 3824.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3824.306 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1080.746 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3824.306 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [154, 54, 197]
}

# Adjusted More
B450x300x18x36_ = {
     'name': '450x300x18x36',
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 18,
     'A(cm2)': 284.04,
     'J(cm4)': 964.967,
     'I_y(cm4)': 16218.371,
     'I_z(cm4)': 100888.643,
     'S_y(cm3)': 1081.225,
     'S_z(cm3)': 4483.940,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 4483.940 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4483.940 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1081.225 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 4483.940 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 49, 179]
}

B450x300x18x40_ = {
     'name': '450x300x18x40',
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 40,
     't_w(mm)': 18,
     'A(cm2)': 306.60,
     'J(cm4)': 1279.350,
     'I_y(cm4)': 18017.982,
     'I_z(cm4)': 108777.950,
     'S_y(cm3)': 1201.199,
     'S_z(cm3)': 4834.576,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (40/10)),
     'My_z(kN-mm)': 4834.576 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4834.576 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1201.199 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 4834.576 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [134, 44, 161]
}

B450x300x18x45_ = {
     'name': '450x300x18x45',
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 45,
     't_w(mm)': 18,
     'A(cm2)': 334.80,
     'J(cm4)': 1764.220,
     'I_y(cm4)': 20267.496,
     'I_z(cm4)': 118170.900,
     'S_y(cm3)': 1351.166,
     'S_z(cm3)': 5252.040,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (45/10)),
     'My_z(kN-mm)': 5252.040 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5252.040 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1351.166 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 5252.040 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 39, 143]
}

B500x350x20x40_ = {
     'name': '500x350x20x40',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 40,
     't_w(mm)': 20,
     'A(cm2)': 364.00,
     'J(cm4)': 1544.040,
     'I_y(cm4)': 28611.333,
     'I_z(cm4)': 160841.333,
     'S_y(cm3)': 1634.933,
     'S_z(cm3)': 6433.653,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (40/10)),
     'My_z(kN-mm)': 6433.653 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6433.653 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1634.933 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 6433.653 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [112, 34, 125]
}

B500x350x20x45_ = {
     'name': '500x350x20x45',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 20,
     'A(cm2)': 397.00,
     'J(cm4)': 2118.130,
     'I_y(cm4)': 32183.583,
     'I_z(cm4)': 175050.583,
     'S_y(cm3)': 1839.062,
     'S_z(cm3)': 7002.023,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7002.023 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7002.023 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.062 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7002.023 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [101, 29, 107]
}

B500x350x20x50_ = {
     'name': '500x350x20x50',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 50,
     't_w(mm)': 20,
     'A(cm2)': 430.00,
     'J(cm4)': 2828.310,
     'I_y(cm4)': 35755.833,
     'I_z(cm4)': 188583.333,
     'S_y(cm3)': 2043.190,
     'S_z(cm3)': 7543.333,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (50/10)),
     'My_z(kN-mm)': 7543.333 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7543.333 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2043.190 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7543.333 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
}

# More
B500x350x16x40 = {
     'name': '500x350x16x40',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 40,
     't_w(mm)': 16,
     'A(cm2)': 347.20,
     'J(cm4)': 1475.580,
     'I_y(cm4)': 28597.669,
     'I_z(cm4)': 158731.733,
     'S_y(cm3)': 1634.153,
     'S_z(cm3)': 6334.869,
     'Av_y(cm2)': (500/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (40/10)),
     'My_z(kN-mm)': 6334.869 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6334.869 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1634.153 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 6334.869 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 49, 179]
}

B500x350x20x45 = {
     'name': '500x350x20x45',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 20,
     'A(cm2)': 397.00,
     'J(cm4)': 2118.130,
     'I_y(cm4)': 32183.583,
     'I_z(cm4)': 175050.583,
     'S_y(cm3)': 1839.062,
     'S_z(cm3)': 7002.023,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7002.023 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7002.023 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.062 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7002.023 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [134, 44, 161]
}

B500x350x22x45 = {
     'name': '500x350x22x45',
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 22,
     'A(cm2)': 405.20,
     'J(cm4)': 2164.340,
     'I_y(cm4)': 32192.631,
     'I_z(cm4)': 176199.267,
     'S_y(cm3)': 1839.579,
     'S_z(cm3)': 7047.971,
     'Av_y(cm2)': (500/10) * (22/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7047.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7047.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.579 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7047.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 39, 143]
}

B600x350x25x45 = {
     'name': '600x350x25x45',
     'H(mm)': 600,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 25,
     'A(cm2)': 442.50,
     'J(cm4)': 2310.340,
     'I_y(cm4)': 32222.656,
     'I_z(cm4)': 270736.875,
     'S_y(cm3)': 1841.295,
     'S_z(cm3)': 9024.563,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 9024.563 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9024.563 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1841.295 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 9024.563 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [112, 34, 125]
}

B600x350x25x50 = {
     'name': '600x350x25x50',
     'H(mm)': 600,
     'B(mm)': 350,
     't_f(mm)': 50,
     't_w(mm)': 25,
     'A(cm2)': 475.00,
     'J(cm4)': 3021.090,
     'I_y(cm4)': 35794.271,
     'I_z(cm4)': 291458.333,
     'S_y(cm3)': 2045.387,
     'S_z(cm3)': 9715.278,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (50/10)),
     'My_z(kN-mm)': 9715.278 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9715.278 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2045.387 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 9715.278 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [101, 29, 107]
}

B600x400x25x50 = {
     'name': '600x400x25x50',
     'H(mm)': 600,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 25,
     'A(cm2)': 525.00,
     'J(cm4)': 3437.370,
     'I_y(cm4)': 53398.438,
     'I_z(cm4)': 329375.000,
     'S_y(cm3)': 2669.922,
     'S_z(cm3)': 10979.167,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (400/10) * (50/10)),
     'My_z(kN-mm)': 10979.167 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 10979.167 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2669.922 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 10979.167 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
}


BEAM_SECTION_DATASET = [
     B350x200x9x16,  B350x200x9x20,  B350x200x12x22,
     B375x200x12x25, B375x250x12x25, B375x250x12x28,
     B400x250x12x32, B400x250x16x36, B400x300x16x36, 

     B450x300x18x36_, B450x300x18x40_, B450x300x18x45_,
     B500x350x20x40_, B500x350x20x45_, B500x350x20x50_,

     B500x350x16x40, B500x350x20x45, B500x350x22x45, 
     B600x350x25x45, B600x350x25x50, B600x400x25x50
]

beam_sections = BEAM_SECTION_DATASET[0:15]  # original - 0:9


# Various column sections
# Original
C350x350x20 = {
     'name': '350x350x20',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 20,
     't_w(mm)': 20,
     'A(cm2)': 264.00,
     'J(cm4)': 74216.300,
     'I_y(cm4)': 48092.000,
     'I_z(cm4)': 48092.000,
     'S_y(cm3)': 2748.114,
     'S_z(cm3)': 2748.114,
     'Av_y(cm2)': 2 * (350/10) * (20/10),
     'Av_z(cm2)': 2 * (350/10) * (20/10),
     'My_z(kN-mm)': 2748.114 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2748.114 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2748.114 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [237, 245, 80]
}

C350x350x22 = {
     'name': '350x350x22',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 22,
     't_w(mm)': 22,
     'A(cm2)': 288.64,
     'J(cm4)': 80446.400,
     'I_y(cm4)': 51987.912,
     'I_z(cm4)': 51987.912,
     'S_y(cm3)': 2970.737,
     'S_z(cm3)': 2970.737,
     'Av_y(cm2)': 2 * (350/10) * (22/10),
     'Av_z(cm2)': 2 * (350/10) * (22/10),
     'My_z(kN-mm)': 2970.737 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2970.737 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2970.737 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [223, 236, 76]
}

C350x350x25 = {
     'name': '350x350x25',
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 25,
     't_w(mm)': 25,
     'A(cm2)': 325.00,
     'J(cm4)': 89268.300,
     'I_y(cm4)': 57552.083,
     'I_z(cm4)': 57552.083,
     'S_y(cm3)': 3288.690,
     'S_z(cm3)': 3288.690,
     'Av_y(cm2)': 2 * (350/10) * (25/10),
     'Av_z(cm2)': 2 * (350/10) * (25/10),
     'My_z(kN-mm)': 3288.690 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3288.690 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 3288.690 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [210, 227, 73]
}

C375x375x28 = {
     'name': '375x375x28',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 28,
     't_w(mm)': 28,
     'A(cm2)': 388.64,
     'J(cm4)': 121929.000,
     'I_y(cm4)': 78550.745,
     'I_z(cm4)': 78550.745,
     'S_y(cm3)': 4186.706,
     'S_z(cm3)': 4186.706,
     'Av_y(cm2)': 2 * (375/10) * (28/10),
     'Av_z(cm2)': 2 * (375/10) * (28/10),
     'My_z(kN-mm)': 4186.706 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4186.706 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4186.706 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [197, 218, 69]
}

C375x375x32 = {
     'name': '375x375x32',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 32,
     't_w(mm)': 32,
     'A(cm2)': 439.039,
     'J(cm4)': 135506.000,
     'I_y(cm4)': 86836.989,
     'I_z(cm4)': 86836.989,
     'S_y(cm3)': 4631.306,
     'S_z(cm3)': 4631.306,
     'Av_y(cm2)': 2 * (375/10) * (32/10),
     'Av_z(cm2)': 2 * (375/10) * (32/10),
     'My_z(kN-mm)': 4631.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4631.306 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4631.306 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [184, 210, 66]
}

C375x375x36 = {
     'name': '375x375x36',
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 36,
     't_w(mm)': 36,
     'A(cm2)': 488.159,
     'J(cm4)': 148281.000,
     'I_y(cm4)': 94554.151,
     'I_z(cm4)': 94554.151,
     'S_y(cm3)': 5042.888,
     'S_z(cm3)': 5042.888,
     'Av_y(cm2)': 2 * (375/10) * (36/10),
     'Av_z(cm2)': 2 * (375/10) * (36/10),
     'My_z(kN-mm)': 5042.888 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5042.888 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 5042.888 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [171, 201, 62]
}

C400x400x40 = {
     'name': '400x400x40',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 576.00,
     'J(cm4)': 197757.000,
     'I_y(cm4)': 125951.999,
     'I_z(cm4)': 125951.999,
     'S_y(cm3)': 6297.599,
     'S_z(cm3)': 6297.599,
     'Av_y(cm2)': 2 * (400/10) * (40/10),
     'Av_z(cm2)': 2 * (400/10) * (40/10),
     'My_z(kN-mm)': 6297.599 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6297.599 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6297.599 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [158, 192, 59]
}

C400x400x45 = {
     'name': '400x400x45',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 639.00,
     'J(cm4)': 215326.000,
     'I_y(cm4)': 136373.250,
     'I_z(cm4)': 136373.250,
     'S_y(cm3)': 6818.662,
     'S_z(cm3)': 6818.662,
     'Av_y(cm2)': 2 * (400/10) * (45/10),
     'Av_z(cm2)': 2 * (400/10) * (45/10),
     'My_z(kN-mm)': 6818.662 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6818.662 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6818.662 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 184, 56]
}

C400x400x50 = {
     'name': '400x400x50',
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 700.00,
     'J(cm4)': 231386.000,
     'I_y(cm4)': 145833.333,
     'I_z(cm4)': 145833.333,
     'S_y(cm3)': 7291.666,
     'S_z(cm3)': 7291.666,
     'Av_y(cm2)': 2 * (400/10) * (50/10),
     'Av_z(cm2)': 2 * (400/10) * (50/10),
     'My_z(kN-mm)': 7291.666 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7291.666 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 7291.666 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [132, 175, 52]
}

# Adjusted More
C450x450x46_ = {
     'name': '450x450x46',
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 46,
     't_w(mm)': 46,
     'A(cm2)': 743.36,
     'J(cm4)': 321910.000,
     'I_y(cm4)': 204835.326,
     'I_z(cm4)': 204835.326,
     'S_y(cm3)': 9103.792,
     'S_z(cm3)': 9103.792,
     'Av_y(cm2)': 2 * (450/10) * (46/10),
     'Av_z(cm2)': 2 * (450/10) * (46/10),
     'My_z(kN-mm)': 9103.792 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9103.792 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9103.792 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [119, 166, 49]
}

C450x450x48_ = {
     'name': '450x450x48',
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 48,
     't_w(mm)': 48,
     'A(cm2)': 771.84,
     'J(cm4)': 332050.000,
     'I_y(cm4)': 210851.251,
     'I_z(cm4)': 210851.251,
     'S_y(cm3)': 9371.167,
     'S_z(cm3)': 9371.167,
     'Av_y(cm2)': 2 * (450/10) * (48/10),
     'Av_z(cm2)': 2 * (450/10) * (48/10),
     'My_z(kN-mm)': 9371.167 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9371.167 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9371.167 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [106, 157, 45]
}

C450x450x50_ = {
     'name': '450x450x50',
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 800.00,
     'J(cm4)': 341911.000,
     'I_y(cm4)': 216666.667,
     'I_z(cm4)': 216666.667,
     'S_y(cm3)': 9629.630,
     'S_z(cm3)': 9629.630,
     'Av_y(cm2)': 2 * (450/10) * (50/10),
     'Av_z(cm2)': 2 * (450/10) * (50/10),
     'My_z(kN-mm)': 9629.630 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9629.630 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9629.630 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [93, 149, 42]
}

C500x500x46_ = {
     'name': '500x500x46',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 46,
     't_w(mm)': 46,
     'A(cm2)': 835.36,
     'J(cm4)': 453788.000,
     'I_y(cm4)': 289914.473,
     'I_z(cm4)': 289914.473,
     'S_y(cm3)': 11596.579,
     'S_z(cm3)': 11596.579,
     'Av_y(cm2)': 2 * (500/10) * (46/10),
     'Av_z(cm2)': 2 * (500/10) * (46/10),
     'My_z(kN-mm)': 11596.579 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11596.579 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11596.579 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [80, 140, 38]
}

C500x500x48_ = {
     'name': '500x500x48',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 48,
     't_w(mm)': 48,
     'A(cm2)': 867.84,
     'J(cm4)': 468494.000,
     'I_y(cm4)': 298837.811,
     'I_z(cm4)': 298837.811,
     'S_y(cm3)': 11953.512,
     'S_z(cm3)': 11953.512,
     'Av_y(cm2)': 2 * (500/10) * (48/10),
     'Av_z(cm2)': 2 * (500/10) * (48/10),
     'My_z(kN-mm)': 11953.512 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11953.512 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11953.512 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [67, 131, 35]
}

C500x500x50_ = {
     'name': '500x500x50',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 900.00,
     'J(cm4)': 482806.000,
     'I_y(cm4)': 307500.000,
     'I_z(cm4)': 307500.000,
     'S_y(cm3)': 12300.000,
     'S_z(cm3)': 12300.000,
     'Av_y(cm2)': 2 * (500/10) * (50/10),
     'Av_z(cm2)': 2 * (500/10) * (50/10),
     'My_z(kN-mm)': 12300.000 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 12300.000 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 12300.000 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
}

# More
C500x500x40 = {
     'name': '500x500x40',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 736.00,
     'J(cm4)': 407726.000,
     'I_y(cm4)': 261525.333,
     'I_z(cm4)': 261525.333,
     'S_y(cm3)': 10461.013,
     'S_z(cm3)': 10461.013,
     'Av_y(cm2)': 2 * (500/10) * (40/10),
     'Av_z(cm2)': 2 * (500/10) * (40/10),
     'My_z(kN-mm)': 10461.013 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 10461.013 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 10461.013 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [119, 166, 49]
}

C500x500x45 = {
     'name': '500x500x45',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 819.00,
     'J(cm4)': 446151.000,
     'I_y(cm4)': 285353.250,
     'I_z(cm4)': 285353.250,
     'S_y(cm3)': 11414.130,
     'S_z(cm3)': 11414.130,
     'Av_y(cm2)': 2 * (500/10) * (45/10),
     'Av_z(cm2)': 2 * (500/10) * (45/10),
     'My_z(kN-mm)': 11414.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11414.130 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11414.130 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [106, 157, 45]
}

C500x500x50 = {
     'name': '500x500x50',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 900.00,
     'J(cm4)': 482806.000,
     'I_y(cm4)': 307500.000,
     'I_z(cm4)': 307500.000,
     'S_y(cm3)': 12300.000,
     'S_z(cm3)': 12300.000,
     'Av_y(cm2)': 2 * (500/10) * (50/10),
     'Av_z(cm2)': 2 * (500/10) * (50/10),
     'My_z(kN-mm)': 12300.000 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 12300.000 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 12300.000 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [93, 149, 42]
}

C600x600x40 = {
     'name': '600x600x40',
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 896.00,
     'J(cm4)': 729132.000,
     'I_y(cm4)': 470698.667,
     'I_z(cm4)': 470698.667,
     'S_y(cm3)': 15689.956,
     'S_z(cm3)': 15689.956,
     'Av_y(cm2)': 2 * (600/10) * (40/10),
     'Av_z(cm2)': 2 * (600/10) * (40/10),
     'My_z(kN-mm)': 15689.956 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 15689.956 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 15689.956 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [80, 140, 38]
}

C600x600x45 = {
     'name': '600x600x45',
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 999.00,
     'J(cm4)': 801938.000,
     'I_y(cm4)': 516233.250,
     'I_z(cm4)': 516233.250,
     'S_y(cm3)': 17207.775,
     'S_z(cm3)': 17207.775,
     'Av_y(cm2)': 2 * (600/10) * (45/10),
     'Av_z(cm2)': 2 * (600/10) * (45/10),
     'My_z(kN-mm)': 17207.775 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 17207.775 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 17207.775 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [67, 131, 35]
}

C600x600x50 = {
     'name': '600x600x50',
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 1100.00,
     'J(cm4)': 871829.000,
     'I_y(cm4)': 559166.667,
     'I_z(cm4)': 559166.667,
     'S_y(cm3)': 18638.889,
     'S_z(cm3)': 18638.889,
     'Av_y(cm2)': 2 * (600/10) * (50/10),
     'Av_z(cm2)': 2 * (600/10) * (50/10),
     'My_z(kN-mm)': 18638.889 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 18638.889 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 18638.889 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
}


COLUMN_SECTION_DATASET = [
     C350x350x20, C350x350x22, C350x350x25,
     C375x375x28, C375x375x32, C375x375x36,
     C400x400x40, C400x400x45, C400x400x50,

     C450x450x46_, C450x450x48_, C450x450x50_,
     C500x500x46_, C500x500x48_, C500x500x50_,

     C500x500x40, C500x500x45, C500x500x50, 
     C600x600x40, C600x600x45, C600x600x50,
]

column_sections = COLUMN_SECTION_DATASET[0:15]  # original - 0:9




"""
-----------------------------------------------------------------------------------------------------------------
"""




beam_section_dict = {}
column_section_dict = {}


# Various beam sections
# Original
beam_section_dict['350x200x9x16'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 16,
     't_w(mm)': 9,
     'A(cm2)': 92.62,
     'J(cm4)': 61.836,
     'I_y(cm4)': 2135.265,
     'I_z(cm4)': 20274.4207,
     'S_y(cm3)': 213.526,
     'S_z(cm3)': 1158.538,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (16/10)),
     'My_z(kN-mm)': 1158.538 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1158.538 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 213.526 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1158.538 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [157, 227, 245]
}

beam_section_dict['350x200x9x20'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 20,
     't_w(mm)': 9,
     'A(cm2)': 107.9,
     'J(cm4)': 110.729,
     'I_y(cm4)': 2668.549,
     'I_z(cm4)': 24040.991,
     'S_y(cm3)': 266.855,
     'S_z(cm3)': 1373.770,
     'Av_y(cm2)': (350/10) * (9/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (20/10)),
     'My_z(kN-mm)': 1373.770 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1373.770 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 266.855 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1373.770 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [139, 196, 240]
}

beam_section_dict['350x200x12x22'] = {
     'H(mm)': 350,
     'B(mm)': 200,
     't_f(mm)': 22,
     't_w(mm)': 12,
     'A(cm2)': 124.72,
     'J(cm4)': 155.109,
     'I_y(cm4)': 2937.739,
     'I_z(cm4)': 26569.234,
     'S_y(cm3)': 293.773,
     'S_z(cm3)': 1518.241,
     'Av_y(cm2)': (350/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (22/10)),
     'My_z(kN-mm)': 1518.241 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1518.241 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 293.773 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1518.241 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [122, 165, 235]
}

beam_section_dict['375x200x12x25'] = {
     'H(mm)': 375,
     'B(mm)': 200,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 139.00,
     'J(cm4)': 217.388,
     'I_y(cm4)': 3338.013,
     'I_z(cm4)': 34109.895,
     'S_y(cm3)': 333.801,
     'S_z(cm3)': 1819.194,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (200/10) * (25/10)),
     'My_z(kN-mm)': 1819.194 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 1819.194 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 333.801 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 1819.194 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [105, 134, 230]
}

beam_section_dict['375x250x12x25'] = {
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 25,
     't_w(mm)': 12,
     'A(cm2)': 164.00,
     'J(cm4)': 269.773,
     'I_y(cm4)': 6515.096,
     'I_z(cm4)': 41779.166,
     'S_y(cm3)': 521.207,
     'S_z(cm3)': 2228.222,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (25/10)),
     'My_z(kN-mm)': 2228.222 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2228.222 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 521.207 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2228.222 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [88, 104, 225]
}

beam_section_dict['375x250x12x28'] = {
     'H(mm)': 375,
     'B(mm)': 250,
     't_f(mm)': 28,
     't_w(mm)': 12,
     'A(cm2)': 178.28,
     'J(cm4)': 367.004,
     'I_y(cm4)': 7296.260,
     'I_z(cm4)': 45480.792,
     'S_y(cm3)': 583.700,
     'S_z(cm3)': 2425.700,
     'Av_y(cm2)': (375/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (28/10)),
     'My_z(kN-mm)': 2425.700 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2425.700 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 583.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2425.700 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [71, 73, 220]
}

beam_section_dict['400x250x12x32'] = {
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 32,
     't_w(mm)': 12,
     'A(cm2)': 200.32,
     'J(cm4)': 532.235,
     'I_y(cm4)': 8338.171,
     'I_z(cm4)': 58099.438,
     'S_y(cm3)': 667.053,
     'S_z(cm3)': 2904.971,
     'Av_y(cm2)': (400/10) * (12/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (32/10)),
     'My_z(kN-mm)': 2904.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2904.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 667.053 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 2904.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 42, 215]
}

beam_section_dict['400x250x16x36'] = {
     'H(mm)': 400,
     'B(mm)': 250,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 232.48,
     'J(cm4)': 774.870,
     'I_y(cm4)': 9386.195,
     'I_z(cm4)': 64522.606,
     'S_y(cm3)': 750.895,
     'S_z(cm3)': 3226.130,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (250/10) * (36/10)),
     'My_z(kN-mm)': 3226.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3226.130 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 750.895 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3226.130 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [37, 12, 211]
}

beam_section_dict['400x300x16x36'] = {
     'H(mm)': 400,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 16,
     'A(cm2)': 268.48,
     'J(cm4)': 931.655,
     'I_y(cm4)': 16211.195,
     'I_z(cm4)': 76486.126,
     'S_y(cm3)': 1080.746,
     'S_z(cm3)': 3824.306,
     'Av_y(cm2)': (400/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 3824.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3824.306 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1080.746 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 3824.306 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [154, 54, 197]
}

# Adjusted More
beam_section_dict['450x300x18x36'] = {
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 36,
     't_w(mm)': 18,
     'A(cm2)': 284.04,
     'J(cm4)': 964.967,
     'I_y(cm4)': 16218.371,
     'I_z(cm4)': 100888.643,
     'S_y(cm3)': 1081.225,
     'S_z(cm3)': 4483.940,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (36/10)),
     'My_z(kN-mm)': 4483.940 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4483.940 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1081.225 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 4483.940 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 49, 179]
}

beam_section_dict['450x300x18x40'] = {
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 40,
     't_w(mm)': 18,
     'A(cm2)': 306.60,
     'J(cm4)': 1279.350,
     'I_y(cm4)': 18017.982,
     'I_z(cm4)': 108777.950,
     'S_y(cm3)': 1201.199,
     'S_z(cm3)': 4834.576,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (40/10)),
     'My_z(kN-mm)': 4834.576 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4834.576 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1201.199 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 4834.576 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [134, 44, 161]
}

beam_section_dict['450x300x18x45'] = {
     'H(mm)': 450,
     'B(mm)': 300,
     't_f(mm)': 45,
     't_w(mm)': 18,
     'A(cm2)': 334.80,
     'J(cm4)': 1764.220,
     'I_y(cm4)': 20267.496,
     'I_z(cm4)': 118170.900,
     'S_y(cm3)': 1351.166,
     'S_z(cm3)': 5252.040,
     'Av_y(cm2)': (450/10) * (18/10),
     'Av_z(cm2)': 5/6 * (2 * (300/10) * (45/10)),
     'My_z(kN-mm)': 5252.040 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5252.040 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1351.166 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 5252.040 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 39, 143]
}

beam_section_dict['500x350x20x40'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 40,
     't_w(mm)': 20,
     'A(cm2)': 364.00,
     'J(cm4)': 1544.040,
     'I_y(cm4)': 28611.333,
     'I_z(cm4)': 160841.333,
     'S_y(cm3)': 1634.933,
     'S_z(cm3)': 6433.653,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (40/10)),
     'My_z(kN-mm)': 6433.653 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6433.653 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1634.933 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 6433.653 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [112, 34, 125]
}

beam_section_dict['500x350x20x45'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 20,
     'A(cm2)': 397.00,
     'J(cm4)': 2118.130,
     'I_y(cm4)': 32183.583,
     'I_z(cm4)': 175050.583,
     'S_y(cm3)': 1839.062,
     'S_z(cm3)': 7002.023,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7002.023 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7002.023 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.062 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7002.023 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [101, 29, 107]
}

beam_section_dict['500x350x20x50'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 50,
     't_w(mm)': 20,
     'A(cm2)': 430.00,
     'J(cm4)': 2828.310,
     'I_y(cm4)': 35755.833,
     'I_z(cm4)': 188583.333,
     'S_y(cm3)': 2043.190,
     'S_z(cm3)': 7543.333,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (50/10)),
     'My_z(kN-mm)': 7543.333 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7543.333 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2043.190 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7543.333 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
}

# More
'''
beam_section_dict['500x350x16x40'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 40,
     't_w(mm)': 16,
     'A(cm2)': 347.20,
     'J(cm4)': 1475.580,
     'I_y(cm4)': 28597.669,
     'I_z(cm4)': 158731.733,
     'S_y(cm3)': 1634.153,
     'S_z(cm3)': 6334.869,
     'Av_y(cm2)': (500/10) * (16/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (40/10)),
     'My_z(kN-mm)': 6334.869 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6334.869 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1634.153 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 6334.869 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 49, 179]
}

beam_section_dict['500x350x20x45'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 20,
     'A(cm2)': 397.00,
     'J(cm4)': 2118.130,
     'I_y(cm4)': 32183.583,
     'I_z(cm4)': 175050.583,
     'S_y(cm3)': 1839.062,
     'S_z(cm3)': 7002.023,
     'Av_y(cm2)': (500/10) * (20/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7002.023 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7002.023 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.062 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7002.023 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [134, 44, 161]
}

beam_section_dict['500x350x22x45'] = {
     'H(mm)': 500,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 22,
     'A(cm2)': 405.20,
     'J(cm4)': 2164.340,
     'I_y(cm4)': 32192.631,
     'I_z(cm4)': 176199.267,
     'S_y(cm3)': 1839.579,
     'S_z(cm3)': 7047.971,
     'Av_y(cm2)': (500/10) * (22/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 7047.971 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7047.971 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1839.579 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 7047.971 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [123, 39, 143]
}

beam_section_dict['600x350x25x45'] = {
     'H(mm)': 600,
     'B(mm)': 350,
     't_f(mm)': 45,
     't_w(mm)': 25,
     'A(cm2)': 442.50,
     'J(cm4)': 2310.340,
     'I_y(cm4)': 32222.656,
     'I_z(cm4)': 270736.875,
     'S_y(cm3)': 1841.295,
     'S_z(cm3)': 9024.563,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (45/10)),
     'My_z(kN-mm)': 9024.563 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9024.563 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 1841.295 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 9024.563 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [112, 34, 125]
}

beam_section_dict['600x350x25x50'] = {
     'H(mm)': 600,
     'B(mm)': 350,
     't_f(mm)': 50,
     't_w(mm)': 25,
     'A(cm2)': 475.00,
     'J(cm4)': 3021.090,
     'I_y(cm4)': 35794.271,
     'I_z(cm4)': 291458.333,
     'S_y(cm3)': 2045.387,
     'S_z(cm3)': 9715.278,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (350/10) * (50/10)),
     'My_z(kN-mm)': 9715.278 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9715.278 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2045.387 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 9715.278 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [101, 29, 107]
}

beam_section_dict['600x400x25x50'] = {
     'H(mm)': 600,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 25,
     'A(cm2)': 525.00,
     'J(cm4)': 3437.370,
     'I_y(cm4)': 53398.438,
     'I_z(cm4)': 329375.000,
     'S_y(cm3)': 2669.922,
     'S_z(cm3)': 10979.167,
     'Av_y(cm2)': (600/10) * (25/10),
     'Av_z(cm2)': 5/6 * (2 * (400/10) * (50/10)),
     'My_z(kN-mm)': 10979.167 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 10979.167 * BEAM_SHAPE_FACTOR,
     'Mp_y(kN-mm)': 2669.922 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'Mp_z(kN-mm)': 10979.167 * 1e+3 * BEAM_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [90, 24, 90]
}

'''


# Various column sections
# Original
column_section_dict['350x350x20'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 20,
     't_w(mm)': 20,
     'A(cm2)': 264.00,
     'J(cm4)': 74216.300,
     'I_y(cm4)': 48092.000,
     'I_z(cm4)': 48092.000,
     'S_y(cm3)': 2748.114,
     'S_z(cm3)': 2748.114,
     'Av_y(cm2)': 2 * (350/10) * (20/10),
     'Av_z(cm2)': 2 * (350/10) * (20/10),
     'My_z(kN-mm)': 2748.114 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2748.114 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2748.114 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [237, 245, 80]
}

column_section_dict['350x350x22'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 22,
     't_w(mm)': 22,
     'A(cm2)': 288.64,
     'J(cm4)': 80446.400,
     'I_y(cm4)': 51987.912,
     'I_z(cm4)': 51987.912,
     'S_y(cm3)': 2970.737,
     'S_z(cm3)': 2970.737,
     'Av_y(cm2)': 2 * (350/10) * (22/10),
     'Av_z(cm2)': 2 * (350/10) * (22/10),
     'My_z(kN-mm)': 2970.737 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 2970.737 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 2970.737 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [223, 236, 76]
}

column_section_dict['350x350x25'] = {
     'H(mm)': 350,
     'B(mm)': 350,
     't_f(mm)': 25,
     't_w(mm)': 25,
     'A(cm2)': 325.00,
     'J(cm4)': 89268.300,
     'I_y(cm4)': 57552.083,
     'I_z(cm4)': 57552.083,
     'S_y(cm3)': 3288.690,
     'S_z(cm3)': 3288.690,
     'Av_y(cm2)': 2 * (350/10) * (25/10),
     'Av_z(cm2)': 2 * (350/10) * (25/10),
     'My_z(kN-mm)': 3288.690 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 3288.690 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 3288.690 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [210, 227, 73]
}

column_section_dict['375x375x28'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 28,
     't_w(mm)': 28,
     'A(cm2)': 388.64,
     'J(cm4)': 121929.000,
     'I_y(cm4)': 78550.745,
     'I_z(cm4)': 78550.745,
     'S_y(cm3)': 4186.706,
     'S_z(cm3)': 4186.706,
     'Av_y(cm2)': 2 * (375/10) * (28/10),
     'Av_z(cm2)': 2 * (375/10) * (28/10),
     'My_z(kN-mm)': 4186.706 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4186.706 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4186.706 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [197, 218, 69]
}

column_section_dict['375x375x32'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 32,
     't_w(mm)': 32,
     'A(cm2)': 439.039,
     'J(cm4)': 135506.000,
     'I_y(cm4)': 86836.989,
     'I_z(cm4)': 86836.989,
     'S_y(cm3)': 4631.306,
     'S_z(cm3)': 4631.306,
     'Av_y(cm2)': 2 * (375/10) * (32/10),
     'Av_z(cm2)': 2 * (375/10) * (32/10),
     'My_z(kN-mm)': 4631.306 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 4631.306 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 4631.306 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [184, 210, 66]
}

column_section_dict['375x375x36'] = {
     'H(mm)': 375,
     'B(mm)': 375,
     't_f(mm)': 36,
     't_w(mm)': 36,
     'A(cm2)': 488.159,
     'J(cm4)': 148281.000,
     'I_y(cm4)': 94554.151,
     'I_z(cm4)': 94554.151,
     'S_y(cm3)': 5042.888,
     'S_z(cm3)': 5042.888,
     'Av_y(cm2)': 2 * (375/10) * (36/10),
     'Av_z(cm2)': 2 * (375/10) * (36/10),
     'My_z(kN-mm)': 5042.888 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 5042.888 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 5042.888 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [171, 201, 62]
}

column_section_dict['400x400x40'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 576.00,
     'J(cm4)': 197757.000,
     'I_y(cm4)': 125951.999,
     'I_z(cm4)': 125951.999,
     'S_y(cm3)': 6297.599,
     'S_z(cm3)': 6297.599,
     'Av_y(cm2)': 2 * (400/10) * (40/10),
     'Av_z(cm2)': 2 * (400/10) * (40/10),
     'My_z(kN-mm)': 6297.599 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6297.599 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6297.599 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [158, 192, 59]
}

column_section_dict['400x400x45'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 639.00,
     'J(cm4)': 215326.437,
     'I_y(cm4)': 136373.250,
     'I_z(cm4)': 136373.250,
     'S_y(cm3)': 6818.662,
     'S_z(cm3)': 6818.662,
     'Av_y(cm2)': 2 * (400/10) * (45/10),
     'Av_z(cm2)': 2 * (400/10) * (45/10),
     'My_z(kN-mm)': 6818.662 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 6818.662 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 6818.662 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [145, 184, 56]
}

column_section_dict['400x400x50'] = {
     'H(mm)': 400,
     'B(mm)': 400,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 700.00,
     'J(cm4)': 231386.000,
     'I_y(cm4)': 145833.333,
     'I_z(cm4)': 145833.333,
     'S_y(cm3)': 7291.666,
     'S_z(cm3)': 7291.666,
     'Av_y(cm2)': 2 * (400/10) * (50/10),
     'Av_z(cm2)': 2 * (400/10) * (50/10),
     'My_z(kN-mm)': 7291.666 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 7291.666 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 7291.666 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [132, 175, 52]
}

# Adjusted More
column_section_dict['450x450x46'] = {
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 46,
     't_w(mm)': 46,
     'A(cm2)': 743.36,
     'J(cm4)': 321910.000,
     'I_y(cm4)': 204835.326,
     'I_z(cm4)': 204835.326,
     'S_y(cm3)': 9103.792,
     'S_z(cm3)': 9103.792,
     'Av_y(cm2)': 2 * (450/10) * (46/10),
     'Av_z(cm2)': 2 * (450/10) * (46/10),
     'My_z(kN-mm)': 9103.792 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9103.792 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9103.792 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [119, 166, 49]
}

column_section_dict['450x450x48'] = {
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 48,
     't_w(mm)': 48,
     'A(cm2)': 771.84,
     'J(cm4)': 332050.000,
     'I_y(cm4)': 210851.251,
     'I_z(cm4)': 210851.251,
     'S_y(cm3)': 9371.167,
     'S_z(cm3)': 9371.167,
     'Av_y(cm2)': 2 * (450/10) * (48/10),
     'Av_z(cm2)': 2 * (450/10) * (48/10),
     'My_z(kN-mm)': 9371.167 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9371.167 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9371.167 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [106, 157, 45]
}

column_section_dict['450x450x50'] = {
     'H(mm)': 450,
     'B(mm)': 450,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 800.00,
     'J(cm4)': 341911.000,
     'I_y(cm4)': 216666.667,
     'I_z(cm4)': 216666.667,
     'S_y(cm3)': 9629.630,
     'S_z(cm3)': 9629.630,
     'Av_y(cm2)': 2 * (450/10) * (50/10),
     'Av_z(cm2)': 2 * (450/10) * (50/10),
     'My_z(kN-mm)': 9629.630 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 9629.630 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 9629.630 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [93, 149, 42]
}

column_section_dict['500x500x46'] = {
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 46,
     't_w(mm)': 46,
     'A(cm2)': 835.36,
     'J(cm4)': 453788.000,
     'I_y(cm4)': 289914.473,
     'I_z(cm4)': 289914.473,
     'S_y(cm3)': 11596.579,
     'S_z(cm3)': 11596.579,
     'Av_y(cm2)': 2 * (500/10) * (46/10),
     'Av_z(cm2)': 2 * (500/10) * (46/10),
     'My_z(kN-mm)': 11596.579 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11596.579 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11596.579 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [80, 140, 38]
}

column_section_dict['500x500x48'] = {
     'name': '500x500x48',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 48,
     't_w(mm)': 48,
     'A(cm2)': 867.84,
     'J(cm4)': 468494.000,
     'I_y(cm4)': 298837.811,
     'I_z(cm4)': 298837.811,
     'S_y(cm3)': 11953.512,
     'S_z(cm3)': 11953.512,
     'Av_y(cm2)': 2 * (500/10) * (48/10),
     'Av_z(cm2)': 2 * (500/10) * (48/10),
     'My_z(kN-mm)': 11953.512 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11953.512 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11953.512 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [67, 131, 35]
}

column_section_dict['500x500x50'] = {
     'name': '500x500x50',
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 900.00,
     'J(cm4)': 482806.000,
     'I_y(cm4)': 307500.000,
     'I_z(cm4)': 307500.000,
     'S_y(cm3)': 12300.000,
     'S_z(cm3)': 12300.000,
     'Av_y(cm2)': 2 * (500/10) * (50/10),
     'Av_z(cm2)': 2 * (500/10) * (50/10),
     'My_z(kN-mm)': 12300.000 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 12300.000 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 12300.000 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
}


# More
"""
column_section_dict['500x500x40'] = {
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 736.00,
     'J(cm4)': 407726.000,
     'I_y(cm4)': 261525.333,
     'I_z(cm4)': 261525.333,
     'S_y(cm3)': 10461.013,
     'S_z(cm3)': 10461.013,
     'Av_y(cm2)': 2 * (500/10) * (40/10),
     'Av_z(cm2)': 2 * (500/10) * (40/10),
     'My_z(kN-mm)': 10461.013 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 10461.013 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 10461.013 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [119, 166, 49]
}

column_section_dict['500x500x45'] = {
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 819.00,
     'J(cm4)': 446151.000,
     'I_y(cm4)': 285353.250,
     'I_z(cm4)': 285353.250,
     'S_y(cm3)': 11414.130,
     'S_z(cm3)': 11414.130,
     'Av_y(cm2)': 2 * (500/10) * (45/10),
     'Av_z(cm2)': 2 * (500/10) * (45/10),
     'My_z(kN-mm)': 11414.130 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 11414.130 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 11414.130 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [106, 157, 45]
}

column_section_dict['500x500x50'] = {
     'H(mm)': 500,
     'B(mm)': 500,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 900.00,
     'J(cm4)': 482806.000,
     'I_y(cm4)': 307500.000,
     'I_z(cm4)': 307500.000,
     'S_y(cm3)': 12300.000,
     'S_z(cm3)': 12300.000,
     'Av_y(cm2)': 2 * (500/10) * (50/10),
     'Av_z(cm2)': 2 * (500/10) * (50/10),
     'My_z(kN-mm)': 12300.000 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 12300.000 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 12300.000 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [93, 149, 42]
}

column_section_dict['600x600x40'] = {
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 40,
     't_w(mm)': 40,
     'A(cm2)': 896.00,
     'J(cm4)': 729132.000,
     'I_y(cm4)': 470698.667,
     'I_z(cm4)': 470698.667,
     'S_y(cm3)': 15689.956,
     'S_z(cm3)': 15689.956,
     'Av_y(cm2)': 2 * (600/10) * (40/10),
     'Av_z(cm2)': 2 * (600/10) * (40/10),
     'My_z(kN-mm)': 15689.956 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 15689.956 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 15689.956 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [80, 140, 38]
}

column_section_dict['600x600x45'] = {
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 45,
     't_w(mm)': 45,
     'A(cm2)': 999.00,
     'J(cm4)': 801938.000,
     'I_y(cm4)': 516233.250,
     'I_z(cm4)': 516233.250,
     'S_y(cm3)': 17207.775,
     'S_z(cm3)': 17207.775,
     'Av_y(cm2)': 2 * (600/10) * (45/10),
     'Av_z(cm2)': 2 * (600/10) * (45/10),
     'My_z(kN-mm)': 17207.775 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 17207.775 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 17207.775 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [67, 131, 35]
}

column_section_dict['600x600x50'] = {
     'H(mm)': 600,
     'B(mm)': 600,
     't_f(mm)': 50,
     't_w(mm)': 50,
     'A(cm2)': 1100.00,
     'J(cm4)': 871829.000,
     'I_y(cm4)': 559166.667,
     'I_z(cm4)': 559166.667,
     'S_y(cm3)': 18638.889,
     'S_z(cm3)': 18638.889,
     'Av_y(cm2)': 2 * (600/10) * (50/10),
     'Av_z(cm2)': 2 * (600/10) * (50/10),
     'My_z(kN-mm)': 18638.889 * 1e+3 * YIELDING_STRESS,
     'Z_z(cm3)': 18638.889 * COLUMN_SHAPE_FACTOR,
     'Mp_z(kN-mm)': 18638.889 * 1e+3 * COLUMN_SHAPE_FACTOR * YIELDING_STRESS,
     'color': [54, 123, 32]
}
"""


