use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 5773, sign: true });
a.append(FP16x16 { mag: 5795, sign: true });
a.append(FP16x16 { mag: 2363, sign: true });
a.append(FP16x16 { mag: 3449, sign: true });
a.append(FP16x16 { mag: 3924, sign: true });
a.append(FP16x16 { mag: 241, sign: true });
a.append(FP16x16 { mag: 6357, sign: false });
a.append(FP16x16 { mag: 3321, sign: false });
a.append(FP16x16 { mag: 3448, sign: true });
a.append(FP16x16 { mag: 1387, sign: true });
a.append(FP16x16 { mag: 6665, sign: true });
a.append(FP16x16 { mag: 3347, sign: false });
a.append(FP16x16 { mag: 971, sign: true });
a.append(FP16x16 { mag: 5250, sign: true });
a.append(FP16x16 { mag: 1166, sign: false });
a.append(FP16x16 { mag: 3412, sign: false });
a.append(FP16x16 { mag: 2572, sign: true });
a.append(FP16x16 { mag: 6623, sign: true });
a.append(FP16x16 { mag: 1684, sign: false });
a.append(FP16x16 { mag: 4955, sign: false });
a.append(FP16x16 { mag: 2530, sign: false });
a.append(FP16x16 { mag: 3014, sign: false });
a.append(FP16x16 { mag: 115, sign: true });
a.append(FP16x16 { mag: 3277, sign: false });
a.append(FP16x16 { mag: 2746, sign: true });
a.append(FP16x16 { mag: 1079, sign: true });
a.append(FP16x16 { mag: 1445, sign: false });
a.append(FP16x16 { mag: 5300, sign: true });
a.append(FP16x16 { mag: 4, sign: false });
a.append(FP16x16 { mag: 6037, sign: false });
a.append(FP16x16 { mag: 5855, sign: false });
a.append(FP16x16 { mag: 3386, sign: true });
a.append(FP16x16 { mag: 666, sign: true });
a.append(FP16x16 { mag: 7036, sign: false });
a.append(FP16x16 { mag: 1445, sign: true });
a.append(FP16x16 { mag: 2154, sign: true });
a.append(FP16x16 { mag: 589, sign: true });
a.append(FP16x16 { mag: 2051, sign: false });
a.append(FP16x16 { mag: 5587, sign: true });
a.append(FP16x16 { mag: 1667, sign: true });
a.append(FP16x16 { mag: 2324, sign: false });
a.append(FP16x16 { mag: 3647, sign: true });
a.append(FP16x16 { mag: 531, sign: true });
a.append(FP16x16 { mag: 2627, sign: false });
a.append(FP16x16 { mag: 7071, sign: false });
a.append(FP16x16 { mag: 869, sign: true });
a.append(FP16x16 { mag: 5846, sign: true });
a.append(FP16x16 { mag: 2981, sign: false });
a.append(FP16x16 { mag: 1516, sign: false });
a.append(FP16x16 { mag: 4834, sign: true });
a.append(FP16x16 { mag: 2009, sign: false });
a.append(FP16x16 { mag: 3426, sign: true });
a.append(FP16x16 { mag: 6742, sign: true });
a.append(FP16x16 { mag: 3056, sign: true });
a.append(FP16x16 { mag: 3529, sign: false });
a.append(FP16x16 { mag: 4397, sign: true });
a.append(FP16x16 { mag: 4935, sign: true });
a.append(FP16x16 { mag: 1526, sign: true });
a.append(FP16x16 { mag: 46, sign: false });
a.append(FP16x16 { mag: 2818, sign: true });
a.append(FP16x16 { mag: 6171, sign: false });
a.append(FP16x16 { mag: 2502, sign: true });
a.append(FP16x16 { mag: 5479, sign: true });
a.append(FP16x16 { mag: 1804, sign: true });
a.append(FP16x16 { mag: 3326, sign: true });
a.append(FP16x16 { mag: 48, sign: false });
a.append(FP16x16 { mag: 3971, sign: true });
a.append(FP16x16 { mag: 3812, sign: false });
a.append(FP16x16 { mag: 6720, sign: true });
a.append(FP16x16 { mag: 247, sign: false });
a.append(FP16x16 { mag: 6575, sign: true });
a.append(FP16x16 { mag: 1383, sign: false });
a.append(FP16x16 { mag: 881, sign: true });
a.append(FP16x16 { mag: 6402, sign: false });
a.append(FP16x16 { mag: 1893, sign: true });
a.append(FP16x16 { mag: 1183, sign: false });
a.append(FP16x16 { mag: 5858, sign: true });
a.append(FP16x16 { mag: 6253, sign: true });
a.append(FP16x16 { mag: 6544, sign: false });
a.append(FP16x16 { mag: 4665, sign: false });
a.append(FP16x16 { mag: 5363, sign: true });
a.append(FP16x16 { mag: 1734, sign: true });
a.append(FP16x16 { mag: 6025, sign: true });
a.append(FP16x16 { mag: 4824, sign: true });
a.append(FP16x16 { mag: 4124, sign: true });
a.append(FP16x16 { mag: 2842, sign: false });
a.append(FP16x16 { mag: 3683, sign: true });
a.append(FP16x16 { mag: 4181, sign: false });
a.append(FP16x16 { mag: 6005, sign: true });
a.append(FP16x16 { mag: 3641, sign: false });
a.append(FP16x16 { mag: 2355, sign: false });
a.append(FP16x16 { mag: 4674, sign: true });
a.append(FP16x16 { mag: 586, sign: true });
a.append(FP16x16 { mag: 4955, sign: true });
a.append(FP16x16 { mag: 7006, sign: false });
a.append(FP16x16 { mag: 3795, sign: true });
a.append(FP16x16 { mag: 5268, sign: true });
a.append(FP16x16 { mag: 1624, sign: false });
a.append(FP16x16 { mag: 1743, sign: true });
a.append(FP16x16 { mag: 4664, sign: false });
a.append(FP16x16 { mag: 5673, sign: false });
a.append(FP16x16 { mag: 421, sign: false });
a.append(FP16x16 { mag: 3991, sign: true });
a.append(FP16x16 { mag: 6087, sign: false });
a.append(FP16x16 { mag: 6019, sign: true });
a.append(FP16x16 { mag: 5619, sign: true });
a.append(FP16x16 { mag: 588, sign: true });
a.append(FP16x16 { mag: 5070, sign: true });
a.append(FP16x16 { mag: 3248, sign: true });
a.append(FP16x16 { mag: 3113, sign: false });
a.append(FP16x16 { mag: 1525, sign: true });
a.append(FP16x16 { mag: 1617, sign: false });
a.append(FP16x16 { mag: 3930, sign: true });
a.append(FP16x16 { mag: 5314, sign: false });
a.append(FP16x16 { mag: 3290, sign: false });
a.append(FP16x16 { mag: 310, sign: true });
a.append(FP16x16 { mag: 2625, sign: false });
a.append(FP16x16 { mag: 6517, sign: false });
a.append(FP16x16 { mag: 6664, sign: false });
a.append(FP16x16 { mag: 1801, sign: false });
a.append(FP16x16 { mag: 396, sign: true });
a.append(FP16x16 { mag: 6525, sign: false });
a.append(FP16x16 { mag: 5379, sign: true });
a.append(FP16x16 { mag: 6472, sign: false });
a.append(FP16x16 { mag: 4299, sign: true });
a.append(FP16x16 { mag: 2456, sign: true });
a.append(FP16x16 { mag: 4351, sign: false });
a.append(FP16x16 { mag: 108, sign: false });
a.append(FP16x16 { mag: 4695, sign: false });
a.append(FP16x16 { mag: 3155, sign: false });
a.append(FP16x16 { mag: 3751, sign: true });
a.append(FP16x16 { mag: 3089, sign: true });
a.append(FP16x16 { mag: 1436, sign: true });
a.append(FP16x16 { mag: 2828, sign: false });
a.append(FP16x16 { mag: 4214, sign: false });
a.append(FP16x16 { mag: 268, sign: false });
a.append(FP16x16 { mag: 1342, sign: true });
a.append(FP16x16 { mag: 5218, sign: true });
a.append(FP16x16 { mag: 4438, sign: true });
a.append(FP16x16 { mag: 1261, sign: false });
a.append(FP16x16 { mag: 532, sign: false });
a.append(FP16x16 { mag: 5865, sign: false });
a.append(FP16x16 { mag: 4264, sign: true });
a.append(FP16x16 { mag: 1795, sign: false });
a.append(FP16x16 { mag: 367, sign: true });
a.append(FP16x16 { mag: 6463, sign: true });
a.append(FP16x16 { mag: 195, sign: false });
a.append(FP16x16 { mag: 1982, sign: true });
a.append(FP16x16 { mag: 2999, sign: false });
a.append(FP16x16 { mag: 662, sign: true });
a.append(FP16x16 { mag: 4480, sign: true });
a.append(FP16x16 { mag: 6069, sign: false });
a.append(FP16x16 { mag: 145, sign: false });
a.append(FP16x16 { mag: 3527, sign: false });
a.append(FP16x16 { mag: 6018, sign: true });
a.append(FP16x16 { mag: 6354, sign: false });
a.append(FP16x16 { mag: 1006, sign: false });
a.append(FP16x16 { mag: 1816, sign: false });
a.append(FP16x16 { mag: 1333, sign: true });
a.append(FP16x16 { mag: 2445, sign: false });
a.append(FP16x16 { mag: 5977, sign: false });
a.append(FP16x16 { mag: 6779, sign: true });
a.append(FP16x16 { mag: 6207, sign: true });
a.append(FP16x16 { mag: 6277, sign: true });
a.append(FP16x16 { mag: 6683, sign: false });
a.append(FP16x16 { mag: 1168, sign: true });
a.append(FP16x16 { mag: 1935, sign: true });
a.append(FP16x16 { mag: 728, sign: true });
a.append(FP16x16 { mag: 396, sign: true });
a.append(FP16x16 { mag: 1093, sign: false });
a.append(FP16x16 { mag: 3232, sign: false });
a.append(FP16x16 { mag: 977, sign: false });
a.append(FP16x16 { mag: 2293, sign: true });
a.append(FP16x16 { mag: 2510, sign: false });
a.append(FP16x16 { mag: 3208, sign: false });
a.append(FP16x16 { mag: 5116, sign: true });
a.append(FP16x16 { mag: 1681, sign: true });
a.append(FP16x16 { mag: 905, sign: false });
a.append(FP16x16 { mag: 3898, sign: false });
a.append(FP16x16 { mag: 5988, sign: false });
a.append(FP16x16 { mag: 51, sign: false });
a.append(FP16x16 { mag: 4622, sign: true });
a.append(FP16x16 { mag: 5288, sign: false });
a.append(FP16x16 { mag: 6055, sign: false });
a.append(FP16x16 { mag: 4230, sign: false });
a.append(FP16x16 { mag: 4030, sign: true });
a.append(FP16x16 { mag: 1783, sign: true });
a.append(FP16x16 { mag: 3499, sign: false });
a.append(FP16x16 { mag: 104, sign: false });
a.append(FP16x16 { mag: 2902, sign: false });
a.append(FP16x16 { mag: 6535, sign: true });
a.append(FP16x16 { mag: 2554, sign: true });
}