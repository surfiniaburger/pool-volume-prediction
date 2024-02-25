mod chunk0;

use orion::operators::tensor::{FP16x16Tensor, Tensor, TensorTrait};
use orion::numbers::{FixedTrait, FP16x16};

fn get_node_model_conv1d_11_conv1d_expanddims_1_0() -> Tensor<FP16x16> {
    let mut shape = array![1, 32, 1, 1];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}