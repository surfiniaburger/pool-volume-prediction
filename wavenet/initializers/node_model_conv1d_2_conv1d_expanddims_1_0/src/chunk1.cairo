use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 6687, sign: false });
a.append(FP16x16 { mag: 16986, sign: true });
a.append(FP16x16 { mag: 1660, sign: true });
a.append(FP16x16 { mag: 10475, sign: true });
a.append(FP16x16 { mag: 4628, sign: true });
a.append(FP16x16 { mag: 17021, sign: true });
a.append(FP16x16 { mag: 17066, sign: false });
a.append(FP16x16 { mag: 11053, sign: true });
a.append(FP16x16 { mag: 1754, sign: true });
a.append(FP16x16 { mag: 6366, sign: true });
a.append(FP16x16 { mag: 9784, sign: false });
a.append(FP16x16 { mag: 5792, sign: false });
a.append(FP16x16 { mag: 15140, sign: false });
a.append(FP16x16 { mag: 3594, sign: false });
a.append(FP16x16 { mag: 3931, sign: true });
a.append(FP16x16 { mag: 10534, sign: true });
a.append(FP16x16 { mag: 18635, sign: false });
a.append(FP16x16 { mag: 7603, sign: true });
a.append(FP16x16 { mag: 17159, sign: true });
a.append(FP16x16 { mag: 14842, sign: false });
a.append(FP16x16 { mag: 11071, sign: false });
a.append(FP16x16 { mag: 898, sign: true });
a.append(FP16x16 { mag: 11072, sign: true });
a.append(FP16x16 { mag: 12197, sign: false });
}