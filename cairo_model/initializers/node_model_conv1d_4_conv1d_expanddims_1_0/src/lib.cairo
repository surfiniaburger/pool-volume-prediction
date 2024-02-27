mod chunk0;
mod chunk1;

use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_model_conv1d_4_conv1d_expanddims_1_0() -> Tensor<FP16x16> {
    let mut shape = array![32, 32, 1, 1];

    let mut data = array![];
     chunk0::compute(ref data);
     chunk1::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}