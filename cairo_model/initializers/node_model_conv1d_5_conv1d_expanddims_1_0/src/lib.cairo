mod chunk0;
mod chunk1;
mod chunk2;
mod chunk3;
mod chunk4;
mod chunk5;
mod chunk6;
mod chunk7;
mod chunk8;

use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_model_conv1d_5_conv1d_expanddims_1_0() -> Tensor<FP16x16> {
    let mut shape = array![32, 32, 1, 8];

    let mut data = array![];
     chunk0::compute(ref data);
     chunk1::compute(ref data);
     chunk2::compute(ref data);
     chunk3::compute(ref data);
     chunk4::compute(ref data);
     chunk5::compute(ref data);
     chunk6::compute(ref data);
     chunk7::compute(ref data);
     chunk8::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}