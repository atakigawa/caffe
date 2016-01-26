#ifndef CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
#define CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cmath>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpatialTransformerLayer : public Layer<Dtype> {
  public:
    explicit SpatialTransformerLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {
          compute_dU_ = true;
          pre_defined_count = 0;
      }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SpatialTransformer"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  private:
    inline Dtype abs(Dtype x) {
      if (x < 0) return -x; return x;
    }
    inline Dtype max(Dtype x, Dtype y) {
      if (x < y) return y; return x;
    }

    Dtype transform_forward_cpu(const Dtype* pic, Dtype px, Dtype py);
    void transform_backward_cpu(Dtype dV, const Dtype* U, const Dtype px,
      const Dtype py, Dtype* dU, Dtype& dpx, Dtype& dpy);

    string transform_type_;
    string sampler_type_;

    int output_H_;
    int output_W_;

    int N, C, H, W;

    bool compute_dU_;

    Blob<Dtype> dTheta_tmp; // used for back propagation part in GPU implementation
    Blob<Dtype> all_ones; // used for back propagation part in GPU implementation
    Blob<Dtype> full_theta; // used for storing data and diff for full six-dim theta
    Dtype pre_defined_theta[6];
    bool is_pre_defined_theta[6];
    int pre_defined_count;

    // grid that holds collection of (xt, yt, 1) for
    // multiplying with theta. values are [-1, 1].
    Blob<Dtype> grid_t;
    // grid that each element holds corresponding coordinate on input image.
    Blob<Dtype> grid_s;
};

}  // namespace caffe

#endif  // CAFFE_SPATIAL_TRANSFORMER_LAYER_HPP_
