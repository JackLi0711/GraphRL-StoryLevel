'''
beam sections:
H x B x tw x tf
(10) B450x300x18x36_
(11) B450x300x18x40_
(12) B450x300x18x45_
(13) B500x350x20x40_
(14) B500x350x20x45_
(15) B500x350x20x50_

column sections:
Dx x Dy x t
(10) C450x450x46_
(11) C450x450x48_
(12) C450x450x50_
(13) C500x500x46_
(14) C500x500x48_
(15) C500x500x50_
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


# Various column sections
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




# Various column sections
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



