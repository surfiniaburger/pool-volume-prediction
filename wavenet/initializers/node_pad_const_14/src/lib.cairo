mod chunk0;

use orion::operators::tensor::{U32Tensor, Tensor, TensorTrait};


fn get_node_pad_const_14() -> Tensor<u32> {
    let mut shape = array![6];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}