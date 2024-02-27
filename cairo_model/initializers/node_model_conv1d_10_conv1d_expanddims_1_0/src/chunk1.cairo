use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 1456, sign: true });
a.append(FP16x16 { mag: 7134, sign: true });
a.append(FP16x16 { mag: 10126, sign: false });
a.append(FP16x16 { mag: 14795, sign: false });
a.append(FP16x16 { mag: 12385, sign: false });
a.append(FP16x16 { mag: 6170, sign: true });
a.append(FP16x16 { mag: 5120, sign: false });
a.append(FP16x16 { mag: 2881, sign: true });
a.append(FP16x16 { mag: 4989, sign: true });
a.append(FP16x16 { mag: 12672, sign: false });
a.append(FP16x16 { mag: 938, sign: false });
a.append(FP16x16 { mag: 11726, sign: false });
a.append(FP16x16 { mag: 15196, sign: false });
a.append(FP16x16 { mag: 19546, sign: true });
a.append(FP16x16 { mag: 14624, sign: false });
a.append(FP16x16 { mag: 472, sign: false });
a.append(FP16x16 { mag: 14176, sign: false });
a.append(FP16x16 { mag: 10247, sign: false });
a.append(FP16x16 { mag: 3064, sign: true });
a.append(FP16x16 { mag: 5876, sign: true });
a.append(FP16x16 { mag: 13584, sign: true });
a.append(FP16x16 { mag: 7742, sign: true });
a.append(FP16x16 { mag: 2114, sign: false });
a.append(FP16x16 { mag: 15136, sign: true });
}