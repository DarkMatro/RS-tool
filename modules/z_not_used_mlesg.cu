__global__ void arrange(float *mle_range, float *xe, float sigma, int cols)
    {
      const int idx = threadIdx.x + blockDim.x * blockIdx.x;
      int x_id = idx / cols;
      int y_id = idx - x_id * cols;

      float start = xe[x_id] - 3 * sigma;
      float step = 6 * sigma / __int2float_rz(cols);
      mle_range[idx] = start + (step * __int2float_rn(y_id));

    }
;

__global__ void le_mle(float *le_mle_range, float *mle_range, float *x_dash, int cols, float p, float lmbd, float *x, float tmp2_denom)
    {
      const int idx = threadIdx.x + blockDim.x * blockIdx.x;
      int x_id = idx / cols;
      float tmp1 = pow(abs(mle_range[idx] - x_dash[x_id]), p);
      float limit1 = lmbd * tmp1;
      float tmp2_nom = pow(x[x_id] - mle_range[idx], 2);
      float limit2 = tmp2_nom / tmp2_denom;
      le_mle_range[idx] = limit1 + limit2;

    }