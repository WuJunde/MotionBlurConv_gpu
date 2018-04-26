/*
 *use gpu to accomplish the motion blur conv(accomplished by Sun Jian on cpu)
 * compile it by 'mex MotionBlurConv_gpu.cu'
 * then call it by MotionBlurConv_gpu(original_img,blur_kernel_len_map,blur_kernel_ori_map) in matlab
 * Written by Wu Junde,izzy843794947@gmail.com 2018.4
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdlib.h>
#include <math.h>

#define eps 2.2204e-16f
#define sign(n) (n<0?-1.0f:1.0f)
#define ABS(x) (x<0?-x:x)
#define M_PI 3.14159265358979323846
/*
 * Device code
 */
void __global__ MotionBlurConv(int w_img,int h_img,double *X,
                               double *bmag,double *bori,double *R)
{
  /* Perform convolution */
 int len, sx, sy;
 int y0= 69, x0 = 69;
 int w_ker, h_ker, w_ker_half, h_ker_half, offSet_img,offSet_ker, off_x, off_y, l, k, num_cand, ik, k_curr_idx;
 unsigned int dims[2];
 double *src_img, *dst;
 double half, phi, cosphi, xsign, sinphi, x2lastpix,  mag, ori,dist2line, dist2cent, linewdt = 1, sumKernel = 0;
 int i = blockIdx.x*blockDim.x + threadIdx.x;
 int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < w_img && j < h_img) {
         /* Set indicex */
         src_img  = X + j * w_img + i;
         dst  = R + j * w_img + i;

         /* Generate kernel value */
         mag = bmag[j * w_img + i];
         ori = bori[j * w_img + i];
         half = (mag - 1) / 2.0f;
         phi = fmod(ori, 180.0) / 180 * M_PI;

         cosphi = cos(phi);
         sinphi = sin(phi);
         xsign = sign(cosphi);

         double tmp = half*cosphi + linewdt*xsign - mag*eps;
         sx = (floor(ABS(tmp)));
         tmp = half*sinphi + linewdt - mag*eps;
         sy = (floor(ABS(tmp)));


         // if (j == y0 & i == x0)
         // {
         //     printf("%f  %f %f %d %d \n", mag, ori, half, sx, sy);
         //     printf(" %d %d \n", sx, sy);
         //
         //     printf("%f  %f %f %f %f %f\n", half, cosphi, linewdt, xsign, mag, eps);
         //     printf("%f %f \n",half*cosphi + linewdt*xsign - mag*eps, half*sinphi + linewdt - mag*eps);
         // }


         /* Convolution at each location */
         sumKernel = 0;*dst = 0;
         for(l = -sy; l <= sy; l++) // y
         {
             //if (j == y0 & i == x0)
             //    printf(" \n ");
             for(k = -sx; k <= sx; k++)  // x
             {
                   // compute the kernel value at currret location
                   dist2line = l * cosphi + k * sinphi;
                   dist2cent = sqrtf(l * l + k * k);

                   // if (j == y0 & i == x0)
                   // printf(" %d %d %f %f \n ", l, k, dist2line, dist2cent);
                   if (ABS(dist2line) <= linewdt & dist2cent >= half) // if it is the end point
                   {
                      x2lastpix = half - ABS((k + dist2line*sinphi)/cosphi);
                      dist2line = sqrt(pow(dist2line, 2) + pow(x2lastpix, 2));
                      // if (j == 294 & i == 197)
                      //     printf(" %d %d %f %f \n ", l, k, dist2line, x2lastpix);
                   }


                   // if (j == y0 & i == x0)
                   //   printf(" %f  %f %f %f \n", dist2line, linewdt, eps, ABS(dist2line));
                   dist2line = linewdt + eps - ABS(dist2line);

                   if (dist2line<0) dist2line = 0;

                   // compute the convolution result
                   offSet_ker = (l + h_ker_half) * w_ker + k + w_ker_half;
                   off_x = (i + k < w_img & i + k >= 0) ? k : (i + k < 0 ? -i : w_img - 1 - i);
                   off_y = (j + l < h_img & j + l >= 0) ? l : (j + l < 0 ? -j : h_img - 1 - j);
                   offSet_img = (off_y) * w_img + off_x;
                   *dst += dist2line * (*(src_img + offSet_img));  /**/

                   sumKernel += dist2line;

                   // if (j == y0 & i == x0)
                   //  {
                   //     printf("( %d %d:  %f) ", l, k, dist2line);
                   //  }
              }
            }



         if (sumKernel > 0)
         {
             *dst /= sumKernel;

            //if (j == y0 & i == x0)
            //{
            //   printf("  (%f %f) ", sumKernel, *dst);
            //}
         }
     }
}


/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const   *matrix_X, *matrix_bmag, *matrix_bori;
    mxGPUArray    *B;
    double        *X, *bmag, *bori, *R;
    int            w_img, h_img, w_ker, h_ker, w_ker_half, h_ker_half, offSet_img,offSet_ker, off_x, off_y, i, j, l, k, num_cand, ik, k_curr_idx;
    unsigned int   dims[2];
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=3) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    matrix_X = mxGPUCreateFromMxArray(prhs[0]);
    matrix_bmag = mxGPUCreateFromMxArray(prhs[1]);
    matrix_bori = mxGPUCreateFromMxArray(prhs[2]);
    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if (mxGPUGetClassID(matrix_X) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    X = (double  *)(mxGPUGetDataReadOnly(matrix_X));
    bmag = (double  *)(mxGPUGetDataReadOnly(matrix_bmag));
    bori = (double  *)(mxGPUGetDataReadOnly(matrix_bori));

    mwSize const *size= mxGPUGetDimensions(matrix_X);
    w_img = (int)(size[0]);
    h_img = (int)(size[1]);

    /* Choose a reasonably sized number of threads for the block. */  /*now only process the N*N case*/
    dim3 threadsPerBlock(32, 32);     /*when 256 threads 16,when 1024 threads 32,if process all at once w_img,h_img*/
    dim3 blocksPerGrid((w_img + threadsPerBlock.x -1) / threadsPerBlock.x, (h_img + threadsPerBlock.y -1) / threadsPerBlock.y);


    /* Create a GPUArray to hold the result and get its underlying pointer. */
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(matrix_X),
                            mxGPUGetDimensions(matrix_X),
                            mxGPUGetClassID(matrix_X),
                            mxGPUGetComplexity(matrix_X),
                            MX_GPU_DO_NOT_INITIALIZE);
    R = (double *)(mxGPUGetData(B));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */

    MotionBlurConv<<<blocksPerGrid, threadsPerBlock>>>(w_img, h_img, X, bmag, bori ,R);

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(matrix_X);
    mxGPUDestroyGPUArray(matrix_bmag);
    mxGPUDestroyGPUArray(matrix_bori);
    mxGPUDestroyGPUArray(B);
}
