mod chunk0;

use orion::operators::tensor::{U32Tensor, Tensor, TensorTrait};


fn get_node_const_fold_opt_139_148() -> Tensor<u32> {
    let mut shape = array![1];

    let mut data = array![];
     chunk0::compute(ref data);

    TensorTrait::new(shape.span(), data.span())
}