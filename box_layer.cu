#include <algorithm>
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/box_layer.hpp"


namespace caffe {


/* copy only clipped region */
  template <typename Dtype>
  void BoxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top) {

    Forward_cpu(bottom,top);
  
  }

/* copy only clipped region */

  template <typename Dtype>
void BoxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

      Backward_cpu(top, propagate_down, bottom);
  }

  INSTANTIATE_LAYER_GPU_FUNCS(BoxLayer);
}

