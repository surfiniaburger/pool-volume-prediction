mod chunk0;

use orion::operators::tensor::{U32Tensor, Tensor, TensorTrait};


fn get_node_new_shape_119() -> Tensor<u32> {
    let mut shape = array![3];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}