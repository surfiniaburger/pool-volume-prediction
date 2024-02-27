use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 15877, sign: true });
a.append(FP16x16 { mag: 17329, sign: true });
a.append(FP16x16 { mag: 1931, sign: true });
a.append(FP16x16 { mag: 4158, sign: true });
a.append(FP16x16 { mag: 9704, sign: false });
a.append(FP16x16 { mag: 364, sign: false });
a.append(FP16x16 { mag: 5618, sign: false });
a.append(FP16x16 { mag: 4838, sign: false });
a.append(FP16x16 { mag: 17786, sign: false });
a.append(FP16x16 { mag: 10376, sign: false });
a.append(FP16x16 { mag: 6672, sign: false });
a.append(FP16x16 { mag: 15992, sign: false });
a.append(FP16x16 { mag: 7874, sign: true });
a.append(FP16x16 { mag: 5135, sign: true });
a.append(FP16x16 { mag: 1500, sign: true });
a.append(FP16x16 { mag: 4505, sign: true });
a.append(FP16x16 { mag: 6550, sign: true });
a.append(FP16x16 { mag: 2838, sign: false });
a.append(FP16x16 { mag: 17694, sign: false });
a.append(FP16x16 { mag: 5083, sign: true });
a.append(FP16x16 { mag: 10310, sign: false });
a.append(FP16x16 { mag: 9855, sign: false });
a.append(FP16x16 { mag: 11261, sign: true });
a.append(FP16x16 { mag: 12549, sign: false });
}