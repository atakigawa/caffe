#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include "caffe/util/benchmark.hpp"
#include "caffe/layers/spatial_transformer_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_value_to_constant(
    const int nthreads, Dtype value,
    int size, int i, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    dst[index * size + i] = value;
  }
}

template <typename Dtype>
__global__ void copy_values(
    const int nthreads,
    int size_src, int k, const Dtype* src,
    int size_dst, int i, Dtype* dst) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    dst[index * size_dst + i] = src[index * size_src + k];
  }
}

template <typename Dtype>
__global__ void SpatialTransformerForwardGPU(
    const int nthreads, int N, int C,
    int H, int W, int output_H_, int output_W_,
    const Dtype* grid_s_data, const Dtype* U, Dtype* V) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int t = index % output_W_;
    const int s = (index / output_W_) % output_H_;
    const int j = (index / (output_W_ * output_H_)) % C;
    const int i = index / (output_W_ * output_H_ * C);

    const Dtype* coordinates = grid_s_data +
      i * (output_H_ * output_W_ * 2) + s * output_W_ * 2 + t * 2;
    const Dtype px = coordinates[0];
    const Dtype py = coordinates[1];

    // convert back to real values
    const Dtype x = (px + 1) / (Dtype)2. * (W - 1);
    const Dtype y = (py + 1) / (Dtype)2. * (H - 1);
    const Dtype* bottom_data_base = U + i * (C * H * W) + j * (H * W);

    Dtype val = (Dtype)0.;
    int m_min = (floor(y) > 0) ? floor(y) : 0;
    int m_max = (ceil(y) < H - 1) ? ceil(y) : H - 1;
    int n_min = (floor(x) > 0) ? floor(x) : 0;
    int n_max = (ceil(x) < W - 1) ? ceil(x) : W - 1;
    // left top, right top, left bottom, right bottom
    for (int m = m_min; m <= m_max; ++m) {
      for (int n = n_min; n <= n_max; ++n) {
        val += (1 - abs(x - n)) * (1 - abs(y - m)) *
          bottom_data_base[m * W + n];
      }
    }
    V[index] = val;
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  string prefix = "SpatialTransformerLayer::Forward_gpu::\t";

  const Dtype* U = bottom[0]->gpu_data();
  const Dtype* theta = bottom[1]->gpu_data();
  const Dtype* grid_t_data = grid_t.gpu_data();
  Dtype* grid_s_data = grid_s.mutable_gpu_data();
  Dtype* full_theta_data = full_theta.mutable_gpu_data();
  Dtype* V = top[0]->mutable_gpu_data();

  caffe_gpu_set(grid_s.count(), (Dtype)0, grid_s_data);
  caffe_gpu_set(top[0]->count(), (Dtype)0, V);

  // compute full_theta
  int k = 0;
  const int num_threads = N;
  for (int i = 0; i < 6; ++i) {
    if (is_pre_defined_theta[i]) {
      set_value_to_constant<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
                                     CAFFE_CUDA_NUM_THREADS>>>(
          num_threads,
          pre_defined_theta[i],
          6, i, full_theta_data);
    } else {
      copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
                           CAFFE_CUDA_NUM_THREADS>>>(
          num_threads,
          6 - pre_defined_count, k, theta,
          6, i, full_theta_data);
      ++k;
    }
  }

  // compute grid_s. (h*w,3) * (2,3).T -> (h*w,2)
  for (int i = 0; i < N; ++i) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
      output_H_ * output_W_, 2, 3,
      (Dtype)1., grid_t_data, full_theta_data + 6 * i,
      (Dtype)0., grid_s_data + (output_H_ * output_W_ * 2) * i);
  }

  // do image sampling (bilinear)
  const int nthreads = N * C * output_H_ * output_W_;
  SpatialTransformerForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                                        CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, N, C,
      H, W, output_H_, output_W_,
      grid_s_data, U, V);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dTheta(
    const int nthreads, int N, int C,
    int H, int W, int output_H_, int output_W_,
    const Dtype* grid_s_data, const Dtype* dV,
    const Dtype* U, Dtype* dTheta_tmp_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int t = index % output_W_;
    const int s = (index / output_W_) % output_H_;
    const int j = (index / (output_W_ * output_H_)) % C;
    const int i = index / (output_W_ * output_H_ * C);

    const Dtype* coordinates = grid_s_data +
      i * (output_H_ * output_W_ * 2) + s * output_W_ * 2 + t * 2;
    const Dtype px = coordinates[0];
    const Dtype py = coordinates[1];

    // convert back to real values
    const Dtype x = (px + 1) / (Dtype)2. * (W - 1);
    const Dtype y = (py + 1) / (Dtype)2. * (H - 1);

    // calculate dV/dx, dV/dy
    const Dtype top_diff = dV[index];
    const Dtype* bottom_data_base = U + i * (C * H * W) + j * (H * W);
    Dtype dx = (Dtype)0.;
    Dtype dy = (Dtype)0.;
    int m_min = (floor(y) > 0) ? floor(y) : 0;
    int m_max = (ceil(y) < H - 1) ? ceil(y) : H - 1;
    int n_min = (floor(x) > 0) ? floor(x) : 0;
    int n_max = (ceil(x) < W - 1) ? ceil(x) : W - 1;
    int sign_x, sign_y;
    // left top, right top, left bottom, right bottom
    for (int m = m_min; m <= m_max; ++m) {
      for (int n = n_min; n <= n_max; ++n) {
        sign_x = (Dtype(0.) <= Dtype(n - x)) - (Dtype(n - x) < Dtype(0.));
        sign_y = (Dtype(0.) <= Dtype(m - y)) - (Dtype(m - y) < Dtype(0.));
        dx += (1 - abs(y - m)) * bottom_data_base[m * W + n] *
          sign_x * top_diff;
        dy += (1 - abs(x - n)) * bottom_data_base[m * W + n] *
          sign_y * top_diff;
      }
    }

    // calculate dxs/dTheta, dys/dTheta.
    // since
    // 1) xs_orig = Theta * xt, and
    // 2) xs = (xs_orig + 1) / 2. * (W - 1),
    // dxs/dTheta = dxs/dxs_orig * dxs_orig/dTheta, which makes
    // dx / 2. * (W - 1) * xt.
    // xt (and yt for y) will be multiplied outside of this function.
    int idx = j * N * 2 * output_H_ * output_W_ +
      i * 2 * output_H_ * output_W_ + s * output_W_ + t;
    dTheta_tmp_diff[idx] = dx / (Dtype)2. * (output_W_ - 1);
    idx += output_H_ * output_W_;
    dTheta_tmp_diff[idx] = dy / (Dtype)2. * (output_H_ - 1);
  }
}

template <typename Dtype>
__global__ void SpatialTransformerBackwardGPU_dU(
    const int nthreads, const int N, const int C,
    const int H,  const int W, const int output_H_, const int output_W_,
    const Dtype* grid_s_data, const Dtype* dV, Dtype* dU) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int t = index % output_W_;
    const int s = (index / output_W_) % output_H_;
    const int j = (index / (output_W_ * output_H_)) % C;
    const int i = index / (output_W_ * output_H_ * C);

    const Dtype* coordinates = grid_s_data +
      i * (output_H_ * output_W_ * 2) + s * output_W_ * 2 + t * 2;
    const Dtype px = coordinates[0];
    const Dtype py = coordinates[1];

    // convert back to real values
    const Dtype x = (px + 1) / (Dtype)2. * (W - 1);
    const Dtype y = (py + 1) / (Dtype)2. * (H - 1);

    Dtype top_diff = dV[index];
    Dtype* bottom_diff_base = dU + i * (C * H * W) + j * (H * W);
    int m_min = (floor(y) > 0) ? floor(y) : 0;
    int m_max = (ceil(y) < H - 1) ? ceil(y) : H - 1;
    int n_min = (floor(x) > 0) ? floor(x) : 0;
    int n_max = (ceil(x) < W - 1) ? ceil(x) : W - 1;
    // left top, right top, left bottom, right bottom
    for (int m = m_min; m <= m_max; ++m) {
      for (int n = n_min; n <= n_max; ++n) {
        caffe_gpu_atomic_add(
          (1 - abs(x - n)) * (1 - abs(y - m)) * top_diff,
          bottom_diff_base + (m * W + n));
      }
    }
  }
}

template <typename Dtype>
void SpatialTransformerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  string prefix = "SpatialTransformerLayer::Backward_GPU::\t";

  const Dtype* dV = top[0]->gpu_diff();
  const Dtype* grid_s_data = grid_s.gpu_data();
  const Dtype* grid_t_data = grid_t.gpu_data();
  const Dtype* U = bottom[0]->gpu_data();

  Dtype* dTheta_tmp_diff = dTheta_tmp.mutable_gpu_diff();
  caffe_gpu_set(dTheta_tmp.count(), (Dtype)0., dTheta_tmp_diff);
  Dtype* grid_s_diff = grid_s.mutable_gpu_diff();
  Dtype* dFull_theta = full_theta.mutable_gpu_diff();
  Dtype* dTheta = bottom[1]->mutable_gpu_diff();

  const int nthreads = N * C * output_H_ * output_W_;

  SpatialTransformerBackwardGPU_dTheta<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                                                CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, N, C,
      H, W, output_H_, output_W_,
      grid_s_data, dV,
      U, dTheta_tmp_diff);

  //std::cout << "dTheta_tmp_diff start" << std::endl;
  //const Dtype* hogefuga = dTheta_tmp.cpu_diff();
  //for(int i=0; i<C; ++i) {
  //  std::cout<<"channel="<<i<<std::endl;
  //  for(int j=0; j<N; ++j) {
  //    std::cout<<"num="<<j<<std::endl;
  //    for(int k=0; k<(2 * output_H_ * output_W_); ++k) {
  //      std::cout << hogefuga[i*N*output_H_*output_W_*2+j*output_H_*output_W_*2+k] << " ";
  //    }
  //    std::cout<<std::endl;
  //  }
  //  std::cout << "dTheta_tmp_diff end" << std::endl;
  //}

  // (C,N,2*outH*outW).T * (C,) -> (N,2,outH*outW)
  // squash channel dimension here, since adding inside GPU loop
  // requires atomic add.
  Dtype* all_ones_data = all_ones.mutable_gpu_data();
  caffe_gpu_set(all_ones.count(), (Dtype)1., all_ones_data);
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    N * 2 *output_H_ * output_W_, 1, C,
    (Dtype)1., dTheta_tmp_diff, all_ones_data,
    (Dtype)0., grid_s_diff);

  //const Dtype* hogefuga2 = grid_s.cpu_diff();
  //for(int n=0; n<N; ++n) {
  //  std::cout << "grid_s_diff n="<<n<<" start" << std::endl;
  //  for(int i=0; i<2; ++i) {
  //    std::cout << "i="<<i<<std::endl;
  //    for(int j=0; j<(output_H_ * output_W_); ++j) {
  //      std::cout << hogefuga2[n*2*output_H_*output_W_+i*output_H_*output_W_+j] << " ";
  //    }
  //    std::cout<<std::endl;
  //  }
  //  std::cout << "grid_s_diff end" << std::endl;
  //}

  // N times, (2,outH*outW) * (outH*outW,3) -> (2,3)
  for (int i = 0; i < N; ++i) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      2, 3, output_H_ * output_W_,
      (Dtype)1., grid_s_diff + i * (2 * output_H_ * output_W_), grid_t_data,
      (Dtype)0., dFull_theta + i * 6);
  }

  //std::cout << "db_dFull_theta start" << std::endl;
  //const Dtype* db_dFull_theta = full_theta.cpu_diff();
  //for (int i=0; i<full_theta.count(); ++i) {
  //  std::cout << db_dFull_theta[i] << " ";
  //}
  //std::cout<<std::endl;
  //std::cout << "db_dFull_theta end" << std::endl;

  // copy values to dTheta (skip predefined ones)
  int k = 0;
  const int num_threads = N;
  for (int i = 0; i < 6; ++i) {
    if (!is_pre_defined_theta[i]) {
      copy_values<Dtype><<<CAFFE_GET_BLOCKS(num_threads),
                           CAFFE_CUDA_NUM_THREADS>>>(
          num_threads,
          6, i, dFull_theta,
          6 - pre_defined_count, k, dTheta);
      ++k;
    }
  }

  if (compute_dU_) {
    Dtype* dU = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);
    SpatialTransformerBackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
                                              CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, N, C,
        H, W, output_H_, output_W_,
        grid_s_data, dV, dU);
  }

  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SpatialTransformerLayer);

}  // namespace caffe
