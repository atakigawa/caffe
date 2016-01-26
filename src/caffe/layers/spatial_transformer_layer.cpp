#include <vector>

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  string prefix = "\t\tSpatial Transformer Layer:: LayerSetUp: \t";

  if (this->layer_param_.st_param().transform_type() == "affine") {
    transform_type_ = "affine";
  } else {
    CHECK(false) << prefix << \
      "Transformation type only supports affine now!" << std::endl;
  }

  if (this->layer_param_.st_param().sampler_type() == "bilinear") {
    sampler_type_ = "bilinear";
  } else {
    CHECK(false) << prefix << \
      "Sampler type only supports bilinear now!" << std::endl;
  }

  if (this->layer_param_.st_param().compute_du()) {
    compute_dU_ = true;
  } else {
    compute_dU_ = false;
  }

  output_H_ = bottom[0]->shape(2);
  if (this->layer_param_.st_param().has_output_h()) {
    output_H_ = this->layer_param_.st_param().output_h();
  }
  output_W_ = bottom[0]->shape(3);
  if (this->layer_param_.st_param().has_output_w()) {
    output_W_ = this->layer_param_.st_param().output_w();
  }

  std::cout << prefix << "output_H_ = " << output_H_ << \
    ", output_W_ = " << output_W_ << std::endl;
  std::cout << prefix << "Getting pre-defined parameters" <<std::endl;

  is_pre_defined_theta[0] = false;
  if (this->layer_param_.st_param().has_theta_1_1()) {
    is_pre_defined_theta[0] = true;
    ++pre_defined_count;
    pre_defined_theta[0] = this->layer_param_.st_param().theta_1_1();
    std::cout << prefix << "Getting pre-defined theta[1][1] = " \
      << pre_defined_theta[0] << std::endl;
  }

  is_pre_defined_theta[1] = false;
  if (this->layer_param_.st_param().has_theta_1_2()) {
    is_pre_defined_theta[1] = true;
    ++pre_defined_count;
    pre_defined_theta[1] = this->layer_param_.st_param().theta_1_2();
    std::cout << prefix << "Getting pre-defined theta[1][2] = " \
      << pre_defined_theta[1] << std::endl;
  }

  is_pre_defined_theta[2] = false;
  if (this->layer_param_.st_param().has_theta_1_3()) {
    is_pre_defined_theta[2] = true;
    ++pre_defined_count;
    pre_defined_theta[2] = this->layer_param_.st_param().theta_1_3();
    std::cout << prefix << "Getting pre-defined theta[1][3] = " \
      << pre_defined_theta[2] << std::endl;
  }

  is_pre_defined_theta[3] = false;
  if (this->layer_param_.st_param().has_theta_2_1()) {
    is_pre_defined_theta[3] = true;
    ++pre_defined_count;
    pre_defined_theta[3] = this->layer_param_.st_param().theta_2_1();
    std::cout << prefix << "Getting pre-defined theta[2][1] = " \
      << pre_defined_theta[3] << std::endl;
  }

  is_pre_defined_theta[4] = false;
  if (this->layer_param_.st_param().has_theta_2_2()) {
    is_pre_defined_theta[4] = true;
    ++pre_defined_count;
    pre_defined_theta[4] = this->layer_param_.st_param().theta_2_2();
    std::cout << prefix << "Getting pre-defined theta[2][2] = " \
      << pre_defined_theta[4] << std::endl;
  }

  is_pre_defined_theta[5] = false;
  if (this->layer_param_.st_param().has_theta_2_3()) {
    is_pre_defined_theta[5] = true;
    ++pre_defined_count;
    pre_defined_theta[5] = this->layer_param_.st_param().theta_2_3();
    std::cout << prefix << "Getting pre-defined theta[2][3] = " \
      << pre_defined_theta[5] << std::endl;
  }

  CHECK(bottom[1]->count(1) + pre_defined_count == 6) << \
     "The dimension of Theta should be 6. Given " << \
     bottom[1]->count(1) << " + " << pre_defined_count << std::endl;
  CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << \
     "The first dimension of Theta and U should be the same" << std::endl;

  vector<int> grid_t_shape(2);
  grid_t_shape[0] = output_H_ * output_W_;
  grid_t_shape[1] = 3;
  grid_t.Reshape(grid_t_shape);
  // convert to [-1, 1] space
  Dtype* data = grid_t.mutable_cpu_data();
  for (int i = 0; i < output_H_ * output_W_; ++i) {
    data[3 * i] = (i % output_W_) * 1.0 / (output_W_ - 1) * 2 - 1;
    data[3 * i + 1] = (i / output_W_) * 1.0 / (output_H_ - 1) * 2 - 1;
    data[3 * i + 2] = 1;
  }

  vector<int> grid_s_shape(3);
  grid_s_shape[0] = bottom[1]->shape(0);
  grid_s_shape[1] = output_H_ * output_W_;
  grid_s_shape[2] = 2;
  grid_s.Reshape(grid_s_shape);

  std::cout << prefix << "Initialization finished." << std::endl;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  string prefix = "\t\tSpatial Transformer Layer:: Reshape: \t";

  N = bottom[0]->shape(0);
  C = bottom[0]->shape(1);
  H = bottom[0]->shape(2);
  W = bottom[0]->shape(3);

  // reshape V
  vector<int> shape(4);
  shape[0] = N;
  shape[1] = C;
  shape[2] = output_H_;
  shape[3] = output_W_;
  top[0]->Reshape(shape);

  // reshape dTheta_tmp
  vector<int> dTheta_tmp_shape(3);
  dTheta_tmp_shape[0] = C;
  dTheta_tmp_shape[1] = N;
  dTheta_tmp_shape[2] = 2 * output_H_ * output_W_;
  dTheta_tmp.Reshape(dTheta_tmp_shape);

  // init all_ones
  vector<int> all_ones_shape(1);
  all_ones_shape[0] = C;
  all_ones.Reshape(all_ones_shape);

  // reshape full_theta
  vector<int> full_theta_shape(2);
  full_theta_shape[0] = N;
  full_theta_shape[1] = 6;
  full_theta.Reshape(full_theta_shape);
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SpatialTransformerLayer);
#endif

INSTANTIATE_CLASS(SpatialTransformerLayer);
REGISTER_LAYER_CLASS(SpatialTransformer);

}  // namespace caffe
