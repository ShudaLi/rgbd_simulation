/* Copyright (c) 2013 Ankur Handa and Suda Li
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#include<thrust/random.h>
#include<thrust/transform.h>
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include <thrust/random/normal_distribution.h>
//#include "add_kinect_noise.h"
//#include <cutil_math.h>

using namespace cv::cuda;

void launch_add_camera_noise(float3* img_array, float3* noisy_image, const float3& sigma_s, const float3& sigma_c,
                             const unsigned int width, const unsigned int height, float scale);

void launch_add_kinect_noise(float3* points3D,
                             float3* normals3D,
                             float3* noisy_points,
                             const unsigned int stridef4,
                             const unsigned int height,
                             float focal_length,
                             float theta_1,
                             float theta_2,
                             float z1,
                             float z2,
                             float z3);

void launch_colour_from_normals(const GpuMat& normals, GpuMat* colour);
void launch_colour_from_normals(float3* normals,
                                float3* colour,
                                const unsigned int stridef3,
                                const unsigned int height);

void gaussian_shifts(float2* tex_coods,
                     const unsigned int stridef2,
                     const unsigned int height,
                     const float _sigma);
void add_gaussian_shifts( const GpuMat& depth_float_, const GpuMat& gaussian_shift_, GpuMat* depth_shifted_ );
void add_depth_noise_barronCVPR2013(float* depth_copy,
								   const int stridef1,
								   const int height);
//void add_depth_noise_barronCVPR2013( const GpuMat& normal_distribution, GpuMat* depth_copy );

void launch_get_z_coordinate_only(float4* vertex_with_noise,
                                  const unsigned int stridef4,
                                  const unsigned int width,
                                  const unsigned int height,
                                  float* noisy_depth,
                                  const unsigned int stridef1
                                  );

void  launch_convert_depth2png(float* noisy_depth,
                               const unsigned int stridef1,
                               unsigned short* noisy_depth_png,
                               const unsigned int strideu16,
                               const unsigned int width,
                               const unsigned int height);

void generate_smooth_noise(GpuMat* smoothNoise, //iu::ImageGpu_32f_C1
                           GpuMat* baseNoise, //iu::ImageGpu_32f_C1
                           const float samplePeriod,
                           const float sampleFrequency,
                           const unsigned int width,
                           const unsigned int height);

void add_noise2vertex(GpuMat* vertex, //iu::ImageGpu_32f_C4
                      GpuMat* normals,//iu::ImageGpu_32f_C4
                      GpuMat* vertex_with_noise,//iu::ImageGpu_32f_C4
                      GpuMat* perlinNoise);//iu::ImageGpu_32f_C1


//iu::ImageGpu_32f_C1 depth
//iu::ImageGpu_32f_C4 vertex
void convertVerts2Depth(const  GpuMat* vertex, GpuMat* depth, float2 pp, float2 fl);
void convertDepth2Verts(const GpuMat& depth, GpuMat* vertex, float2 pp, float2 fl);
void cudaFastNormalEstimation(const cv::cuda::GpuMat& cvgmPts_, cv::cuda::GpuMat* pcvgmNls_ );
//void computeNormal( GpuMat* vertex, GpuMat* normal);