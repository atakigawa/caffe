#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "fstream"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;

// Reference affine transformer for checking results
// compute coordinates and sample feature map explicitly using loops
template <typename Dtype>
void affine_transform(
    const Blob<Dtype>* in, const Blob<Dtype>* theta,
    Blob<Dtype>* out) {
  int num = in->shape(0);
  int channels = in->shape(1);
  int height = in->shape(2);
  int width = in->shape(3);
  Dtype* out_data = out->mutable_cpu_data();
  caffe_set<Dtype>(out->count(), 0, out_data);
  const Dtype* theta_data = theta->cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < height; ++h) {
      Dtype ty = h / (Dtype)(height - 1) * (Dtype)2. - (Dtype)1.;
      for (int w = 0; w < width; ++w) {
        Dtype tx = w / (Dtype)(width - 1) * (Dtype)2. - (Dtype)1.;
        Dtype sx = tx * theta_data[n * 6] + ty * theta_data[n * 6 + 1] + theta_data[n * 6 + 2];
        Dtype sy = tx * theta_data[n * 6 + 3] + ty * theta_data[n * 6 + 4] + theta_data[n * 6 + 5];
        sx = (sx + 1.) / (Dtype)2. * (width - 1);
        sy = (sy + 1.) / (Dtype)2. * (height - 1);
        for (int c = 0; c < channels; ++c) {
          for (int hh = 0; hh < height; ++hh) {
            for (int ww = 0; ww < width; ++ww) {
              Dtype max_y = 0;
              Dtype max_x = 0;

              max_y = (hh > sy) ? hh - sy : sy - hh;
              max_y = (1 - max_y < 0) ? 0 : 1 - max_y;

              max_x = (ww > sx) ? ww - sx : sx - ww;
              max_x = (1 - max_x < 0) ? 0 : 1 - max_x;

              out_data[out->offset(n, c, h, w)] += in->data_at(n, c, hh, ww) * max_x * max_y;
            }
          }
        }
      }
    }
  }
}
template void affine_transform(
  const Blob<float>* in, const Blob<float>* theta,
  Blob<float>* out);
template void affine_transform(
  const Blob<double>* in, const Blob<double>* theta,
  Blob<double>* out);

// finite difference with mask trick for max operation:
// track the winner (refer to http://cs231n.github.io/neural-networks-3/)
template <typename Dtype>
void theta_gradient(
    const Blob<Dtype>* in, const Blob<Dtype>* theta,
    double delta, Blob<Dtype>* gradient) {
  int num = in->shape(0);
  int channels = in->shape(1);
  int height = in->shape(2);
  int width = in->shape(3);
  Dtype* gradient_data = gradient->mutable_cpu_diff();
  caffe_set<Dtype>(theta->count(), 0, gradient_data);
  const Dtype* theta_data = theta->cpu_data();
  for (int i = 0; i < 6; ++i) {
    for (int n = 0; n < num; ++n) {
      for (int h = 0; h < height; ++h) {
        double ty = h / (double)(height - 1) * (double)2. - (double)1.;
        for (int w = 0; w < width; ++w) {
          double tx = w / (double) (width - 1) * (double)2. - (double)1.;
          double sx = tx * theta_data[n * 6] + ty * theta_data[n * 6 + 1] + theta_data[n * 6 + 2];
          double sy = tx * theta_data[n * 6 + 3] + ty * theta_data[n * 6 + 4] + theta_data[n * 6 + 5];
          double sxn = sx;
          double syn = sy;
          if (i == 0) {
            sxn += delta * tx;
          } else if (i == 1) {
            sxn += delta * ty;
          } else if (i == 2) {
            sxn += delta;
          } else if (i == 3) {
            syn += delta * tx;
          } else if (i == 4) {
            syn += delta * ty;
          } else if (i == 5) {
            syn += delta;
          }
          sx = (sx + 1.) / (double)2. * (width - 1);
          sy = (sy + 1.) / (double)2. * (height - 1);
          sxn = (sxn + 1.) / (double)2. * (width - 1);
          syn = (syn + 1.) / (double)2. * (height - 1);
          for (int c = 0; c < channels; ++c) {
            for (int hh = 0; hh < height; ++hh) {
              for (int ww = 0; ww < width; ++ww) {
                double max_y = 0;
                double max_yn = 0;
                double max_x = 0;
                double max_xn = 0;

                if (hh > sy) {
                  max_y = hh - sy;
                  max_yn = hh - syn;
                } else {
                  max_y = sy - hh;
                  max_yn = syn - hh;
                }
                if (1 - max_y < 0) {
                  max_y = 0;
                  max_yn = 0;
                } else {
                  max_y = 1 - max_y;
                  max_yn = 1 - max_yn;
                }

                if (ww > sx) {
                  max_x = ww - sx;
                  max_xn = ww - sxn;
                } else {
                  max_x = sx - ww;
                  max_xn = sxn - ww;
                }
                if (1 - max_x < 0) {
                  max_x = 0;
                  max_xn = 0;
                } else {
                  max_x = 1 - max_x;
                  max_xn = 1 - max_xn;
                }

                Dtype d = (Dtype)(in->data_at(n, c, hh, ww));
                gradient_data[n * 6 + i] +=
                  (d * max_xn * max_yn - d * max_x * max_y) / delta;
              }
            }
          }
        }
      }
    }
  }
}
template void theta_gradient(
  const Blob<float>* data, const Blob<float>* theta,
  double delta, Blob<float>* gradient);
template void theta_gradient(
  const Blob<double>* data, const Blob<double>* theta,
  double delta, Blob<double>* gradient);


template <typename TypeParam>
class SpatialTransformerLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SpatialTransformerLayerTest()
    : blob_U_(new Blob<Dtype>(2, 3, 5, 9)),
      blob_theta_(new Blob<Dtype>(2, 6, 1, 1)),
      blob_V_(new Blob<Dtype>()),
      blob_V_2(new Blob<Dtype>(2, 3, 3, 3)) {

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_U_);
    filler.Fill(this->blob_theta_);

    blob_bottom_vec_.push_back(blob_U_);
    blob_bottom_vec_.push_back(blob_theta_);
    blob_top_vec_.push_back(blob_V_);
    blob_top_vec_2.push_back(blob_V_2);
  }

  virtual ~SpatialTransformerLayerTest() {
    delete blob_V_2;
    delete blob_V_;
    delete blob_theta_;
    delete blob_U_;
  }

  Blob<Dtype>* blob_U_;
  Blob<Dtype>* blob_theta_;
  Blob<Dtype>* blob_V_;
  Blob<Dtype>* blob_V_2;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_2;
};

TYPED_TEST_CASE(SpatialTransformerLayerTest, TestDtypesGPU);

TYPED_TEST(SpatialTransformerLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
    new SpatialTransformerLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_U_->num(), this->blob_V_->num());
  EXPECT_EQ(this->blob_U_->channels(), this->blob_V_->channels());
  EXPECT_EQ(this->blob_U_->height(), this->blob_V_->height());
  EXPECT_EQ(this->blob_U_->width(), this->blob_V_->width());
}

TYPED_TEST(SpatialTransformerLayerTest, TestIdenticalForward) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  ConstantFiller<Dtype> constant_filler(filler_param);
  constant_filler.Fill(this->blob_theta_);
  this->blob_theta_->mutable_cpu_data()[0] = 1.;
  this->blob_theta_->mutable_cpu_data()[1] = 0.;
  this->blob_theta_->mutable_cpu_data()[2] = 0.;
  this->blob_theta_->mutable_cpu_data()[3] = 0.;
  this->blob_theta_->mutable_cpu_data()[4] = 1.;
  this->blob_theta_->mutable_cpu_data()[5] = 0.;
  this->blob_theta_->mutable_cpu_data()[0 + 6] = 1.;
  this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
  this->blob_theta_->mutable_cpu_data()[2 + 6] = 0.;
  this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
  this->blob_theta_->mutable_cpu_data()[4 + 6] = 1.;
  this->blob_theta_->mutable_cpu_data()[5 + 6] = 0.;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
    new SpatialTransformerLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data;
  const Dtype* bottom_data;
  top_data = this->blob_V_->cpu_data();
  bottom_data = this->blob_U_->cpu_data();
  for (int i = 0; i < this->blob_V_->count(); ++i) {
    EXPECT_NEAR(top_data[i], bottom_data[i], 1e-4);
  }
}

TYPED_TEST(SpatialTransformerLayerTest, TestScalingForward) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  ConstantFiller<Dtype> constant_filler(filler_param);
  constant_filler.Fill(this->blob_theta_);
  this->blob_theta_->mutable_cpu_data()[0] = 2;
  this->blob_theta_->mutable_cpu_data()[1] = 0.;
  this->blob_theta_->mutable_cpu_data()[2] = 1.;
  this->blob_theta_->mutable_cpu_data()[3] = 0.;
  this->blob_theta_->mutable_cpu_data()[4] = 2;
  this->blob_theta_->mutable_cpu_data()[5] = 1.;
  this->blob_theta_->mutable_cpu_data()[0 + 6] = 2;
  this->blob_theta_->mutable_cpu_data()[1 + 6] = 0.;
  this->blob_theta_->mutable_cpu_data()[2 + 6] = 1.;
  this->blob_theta_->mutable_cpu_data()[3 + 6] = 0.;
  this->blob_theta_->mutable_cpu_data()[4 + 6] = 2;
  this->blob_theta_->mutable_cpu_data()[5 + 6] = 1.;
  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
    new SpatialTransformerLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* top_data;
  const Dtype* bottom_data;
  top_data = this->blob_V_->cpu_data();
  bottom_data = this->blob_U_->cpu_data();
  int num = this->blob_V_->num();
  int channels = this->blob_V_->channels();
  int height = this->blob_V_->height();
  int width = this->blob_V_->width();
  for (int n = 0; n < num / 2; ++n) {
    for (int c = 0; c < channels / 2; ++c) {
      for (int h = 0; h < height / 2; ++h) {
        for (int w = 0; w < width / 2; ++w) {
          EXPECT_NEAR(
            top_data[this->blob_V_->offset(n, c, h, w)],
            bottom_data[this->blob_U_->offset(n, c, h * 2, w * 2)],
            1e-4);
        }
      }
    }
  }
}

TYPED_TEST(SpatialTransformerLayerTest, TestAffineForward) {
  typedef typename TypeParam::Dtype Dtype;

  FillerParameter filler_param;
  filler_param.set_min(-1);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  LayerParameter layer_param;
  shared_ptr<Layer<Dtype> > layer(
    new SpatialTransformerLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  shared_ptr<Blob<Dtype> > blob_top_2_;
  blob_top_2_.reset(new Blob<Dtype>());
  blob_top_2_->ReshapeLike(*(this->blob_V_));
  affine_transform(
    this->blob_U_,
    this->blob_theta_,
    blob_top_2_.get());
  const Dtype* top_data = this->blob_V_->cpu_data();
  const Dtype* top_data_2 = blob_top_2_->cpu_data();
  for (int i = 0; i < this->blob_V_->count(); ++i) {
    EXPECT_NEAR(top_data[i], top_data_2[i], 1e-4);
  }
}

// won't work since affine_transform() doesn't allow different output size for now.
//TYPED_TEST(SpatialTransformerLayerTest, TestAffineForwardWithDifferentOutputSize) {
//  typedef typename TypeParam::Dtype Dtype;
//
//  FillerParameter filler_param;
//  filler_param.set_min(-1);
//  filler_param.set_max(1);
//  UniformFiller<Dtype> filler(filler_param);
//  filler.Fill(this->blob_theta_);
//
//  LayerParameter layer_param;
//  SpatialTransformerParameter *st_param = layer_param.mutable_st_param();
//  st_param->set_output_h(3);
//  st_param->set_output_w(3);
//  shared_ptr<Layer<Dtype> > layer(
//    new SpatialTransformerLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_2);
//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_2);
//
//  shared_ptr<Blob<Dtype> > blob_top_2_;
//  blob_top_2_.reset(new Blob<Dtype>());
//  blob_top_2_->ReshapeLike(*(this->blob_V_2));
//  affine_transform(
//    this->blob_U_,
//    this->blob_theta_,
//    blob_top_2_.get());
//  const Dtype* top_data = this->blob_V_2->cpu_data();
//  const Dtype* top_data_2 = blob_top_2_->cpu_data();
//  for (int i = 0; i < this->blob_V_->count(); ++i) {
//    EXPECT_NEAR(top_data[i], top_data_2[i], 1e-4);
//  }
//}

TYPED_TEST(SpatialTransformerLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  // reshape theta to have full 6 dimension
  vector<int> shape_theta(2);
  shape_theta[0] = 2;
  shape_theta[1] = 6;
  this->blob_theta_->Reshape(shape_theta);

  // fill random variables for theta
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  LayerParameter layer_param;
  SpatialTransformerLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(
    &layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SpatialTransformerLayerTest, TestGradientWithDifferentOutputSize) {
  typedef typename TypeParam::Dtype Dtype;

  // reshape theta to have full 6 dimension
  vector<int> shape_theta(2);
  shape_theta[0] = 2;
  shape_theta[1] = 6;
  this->blob_theta_->Reshape(shape_theta);

  // fill random variables for theta
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  // change output size
  LayerParameter layer_param;
  SpatialTransformerParameter *st_param = layer_param.mutable_st_param();
  st_param->set_output_h(3);
  st_param->set_output_w(3);
  SpatialTransformerLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(
    &layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

// test gradients of theta using finite difference method:
// max operator would fail caffe utility
TYPED_TEST(SpatialTransformerLayerTest, TestThetaGradient) {
  typedef typename TypeParam::Dtype Dtype;

  // reshape theta to have full 6 dimension
  vector<int> shape_theta(2);
  shape_theta[0] = 2;
  shape_theta[1] = 6;
  this->blob_theta_->Reshape(shape_theta);

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  LayerParameter layer_param;
  SpatialTransformerLayer<Dtype> layer(layer_param);
  // generate theta_gradient
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_V_->count(); ++i) {
    this->blob_V_->mutable_cpu_diff()[i] = (Dtype)1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  // compute theta gradient using finite difference
  shared_ptr<Blob<Dtype> > blob_theta_2;
  blob_theta_2.reset(new Blob<Dtype>());
  blob_theta_2->ReshapeLike(*(this->blob_theta_));
  theta_gradient(
    this->blob_U_,
    this->blob_theta_,
    (double)1e-4,
    blob_theta_2.get());
  const Dtype* theta_diff = this->blob_theta_->cpu_diff();
  const Dtype* theta_diff_2 = blob_theta_2->cpu_diff();
  for (int i = 0; i < this->blob_theta_->count(); ++i) {
    EXPECT_NEAR(theta_diff[i], theta_diff_2[i], 1e-4) << "i=" << i;
  }
}

TYPED_TEST(SpatialTransformerLayerTest, TestThetaGradientWithDifferentOutputSize) {
  typedef typename TypeParam::Dtype Dtype;

  // reshape theta to have full 6 dimension
  vector<int> shape_theta(2);
  shape_theta[0] = 2;
  shape_theta[1] = 6;
  this->blob_theta_->Reshape(shape_theta);

  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  LayerParameter layer_param;
  SpatialTransformerLayer<Dtype> layer(layer_param);
  SpatialTransformerParameter *st_param = layer_param.mutable_st_param();
  st_param->set_output_h(3);
  st_param->set_output_w(3);
  // generate theta_gradient
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_2);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_2);
  for (int i = 0; i < this->blob_V_2->count(); ++i) {
    this->blob_V_2->mutable_cpu_diff()[i] = (Dtype)1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_2, propagate_down, this->blob_bottom_vec_);

  // compute theta gradient using finite difference
  shared_ptr<Blob<Dtype> > blob_theta_2;
  blob_theta_2.reset(new Blob<Dtype>());
  blob_theta_2->ReshapeLike(*(this->blob_theta_));
  theta_gradient(
    this->blob_U_,
    this->blob_theta_,
    (double)1e-4,
    blob_theta_2.get());
  const Dtype* theta_diff = this->blob_theta_->cpu_diff();
  const Dtype* theta_diff_2 = blob_theta_2->cpu_diff();
  for (int i = 0; i < this->blob_theta_->count(); ++i) {
    EXPECT_NEAR(theta_diff[i], theta_diff_2[i], 1e-4) << "i=" << i;
  }
}

TYPED_TEST(SpatialTransformerLayerTest, TestGradientWithPreDefinedTheta) {
  typedef typename TypeParam::Dtype Dtype;

  int pre_def_count = 2;
  int theta_dim = 6 - pre_def_count;
  // reshape theta to have only pre_def_count dimensions
  vector<int> shape_theta(2);
  shape_theta[0] = 2;
  shape_theta[1] = theta_dim;
  this->blob_theta_->Reshape(shape_theta);

  // fill random variables for theta
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_theta_);

  // set layer_param
  LayerParameter layer_param;
  SpatialTransformerParameter *st_param = layer_param.mutable_st_param();
  st_param->set_theta_1_2(0);
  st_param->set_theta_2_1(0);

  SpatialTransformerLayer<Dtype> layer(layer_param);
  // generate theta_gradient
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_V_->count(); ++i) {
    this->blob_V_->mutable_cpu_diff()[i] = (Dtype)1.;
  }
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  layer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

  // compute theta gradient using finite difference
  // since last dimension of theta is 4 = (6 - 2), create a full
  // theta which has full 6 dimensions.
  Blob<Dtype> blob_theta_2(shape_theta[0], 6, 1, 1);
  shared_ptr<Blob<Dtype> > blob_theta_3;
  blob_theta_3.reset(new Blob<Dtype>(shape_theta[0], 6, 1, 1));
  const Dtype* theta_data = this->blob_theta_->cpu_data();
  Dtype* theta_2_data = blob_theta_2.mutable_cpu_data();
  for (int i = 0; i < shape_theta[0]; ++i) {
    theta_2_data[i * 6 + 0] = theta_data[i * theta_dim + 0];
    theta_2_data[i * 6 + 1] = 0.;
    theta_2_data[i * 6 + 2] = theta_data[i * theta_dim + 1];
    theta_2_data[i * 6 + 3] = 0.;
    theta_2_data[i * 6 + 4] = theta_data[i * theta_dim + 2];
    theta_2_data[i * 6 + 5] = theta_data[i * theta_dim + 3];
  }
  theta_gradient(
    this->blob_U_,
    &blob_theta_2,
    (double)1e-4,
    blob_theta_3.get());
  const Dtype* theta_diff = this->blob_theta_->cpu_diff();
  const Dtype* theta_diff_3 = blob_theta_3->cpu_diff();

  for (int i = 0; i < shape_theta[0]; ++i) {
    // theta_1_1
    EXPECT_NEAR(theta_diff[i * theta_dim + 0], theta_diff_3[i * 6 + 0], 1e-4) << "i=" << i;
    // theta_1_3
    EXPECT_NEAR(theta_diff[i * theta_dim + 1], theta_diff_3[i * 6 + 2], 1e-4) << "i=" << i;
    // theta_2_2
    EXPECT_NEAR(theta_diff[i * theta_dim + 2], theta_diff_3[i * 6 + 4], 1e-4) << "i=" << i;
    // theta_2_3
    EXPECT_NEAR(theta_diff[i * theta_dim + 3], theta_diff_3[i * 6 + 5], 1e-4) << "i=" << i;
  }
}

}  // namespace caffe
