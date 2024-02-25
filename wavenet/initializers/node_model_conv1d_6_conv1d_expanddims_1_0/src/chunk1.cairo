use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 12514, sign: false });
a.append(FP16x16 { mag: 4447, sign: false });
a.append(FP16x16 { mag: 10985, sign: true });
a.append(FP16x16 { mag: 4523, sign: false });
a.append(FP16x16 { mag: 17811, sign: false });
a.append(FP16x16 { mag: 4067, sign: false });
a.append(FP16x16 { mag: 11070, sign: false });
a.append(FP16x16 { mag: 360, sign: false });
a.append(FP16x16 { mag: 3516, sign: false });
a.append(FP16x16 { mag: 11605, sign: false });
a.append(FP16x16 { mag: 9478, sign: false });
a.append(FP16x16 { mag: 19376, sign: false });
a.append(FP16x16 { mag: 9999, sign: false });
a.append(FP16x16 { mag: 13392, sign: true });
a.append(FP16x16 { mag: 3493, sign: false });
a.append(FP16x16 { mag: 992, sign: false });
a.append(FP16x16 { mag: 1488, sign: true });
a.append(FP16x16 { mag: 3610, sign: true });
a.append(FP16x16 { mag: 9685, sign: false });
a.append(FP16x16 { mag: 13174, sign: true });
a.append(FP16x16 { mag: 5658, sign: true });
a.append(FP16x16 { mag: 11761, sign: true });
a.append(FP16x16 { mag: 9337, sign: false });
a.append(FP16x16 { mag: 7503, sign: true });
}