#include<thrust/random.h>
#include<thrust/transform.h>
#include<thrust/device_vector.h>
#include<thrust/iterator/counting_iterator.h>
#include<thrust/iterator/zip_iterator.h>
#include<thrust/tuple.h>
#include <thrust/random/normal_distribution.h>

#include<curand.h>
#include<curand_kernel.h>

#include <boost/math/common_factor_rt.hpp>
#include <assert.h>

#include <opencv2/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "add_kinect_noise.cuh"
#include "vector_math.hpp"

using namespace pcl::device;
using namespace cv::cuda;

typedef thrust::device_vector<float3>::iterator Float3Iterator;
typedef thrust::tuple<Float3Iterator, Float3Iterator> VertexNormalIteratorTuple;
typedef thrust::zip_iterator<VertexNormalIteratorTuple> ZipIterator;
typedef thrust::tuple<float3, float3> VertexNormalTuple;

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

struct ccd_camera_noise
{
    const float sigma_s_red;
    const float sigma_s_green;
    const float sigma_s_blue;

    const float sigma_c_red;
    const float sigma_c_green;
    const float sigma_c_blue;

    const float scale;

    ccd_camera_noise(float _sigma_s_red,
                    float _sigma_s_green,
                    float _sigma_s_blue,
                    float _sigma_c_red,
                    float _sigma_c_green,
                    float _sigma_c_blue,
                    float _scale) : sigma_s_red(_sigma_s_red),
       sigma_s_green(_sigma_s_green),
       sigma_s_blue(_sigma_s_blue),
       sigma_c_red(_sigma_c_red),
       sigma_c_green(_sigma_c_green),
       sigma_c_blue(_sigma_c_blue),
       scale(_scale)
       {}

  __host__ __device__  float3 operator()(const float3& val,
                                         const unsigned int& thread_id)
  {

      float3 noisy_pix;

      clock_t start_time = clock();

      unsigned int seed = hash(thread_id) + start_time;

      thrust::minstd_rand rng(seed);

      noisy_pix.x = val.x/scale;
      noisy_pix.y = val.y/scale;
      noisy_pix.z = val.z/scale;

      thrust::random::normal_distribution<float> red_pnoise  (0.0f,sqrt(val.x)*sigma_s_red  );
      thrust::random::normal_distribution<float> green_pnoise(0.0f,sqrt(val.y)*sigma_s_green);
      thrust::random::normal_distribution<float> blue_pnoise (0.0f,sqrt(val.z)*sigma_s_blue );

      thrust::random::normal_distribution<float> red_cnoise   (0.0f,sigma_c_red  );
      thrust::random::normal_distribution<float> green_cnoise (0.0f,sigma_c_green);
      thrust::random::normal_distribution<float> blue_cnoise  (0.0f,sigma_c_blue );

      noisy_pix.x = noisy_pix.x  + red_pnoise(rng)   + red_cnoise(rng);
      noisy_pix.y = noisy_pix.y  + green_pnoise(rng) + green_cnoise(rng);
      noisy_pix.z = noisy_pix.z  + blue_pnoise(rng)  + blue_cnoise(rng);

      return noisy_pix;
  }
};

void launch_add_camera_noise(float3* img_array, float3* noisy_image, const float3& sigma_s, const float3& sigma_c,
                             const unsigned int width, const unsigned int height, float scale)
{
    thrust::device_ptr<float3>img_src(img_array);
    thrust::device_ptr<float3>img_dest(noisy_image);

    thrust::transform(  img_src, 
						img_src + width*height, 
						thrust::make_counting_iterator(0), 
						img_dest,
						ccd_camera_noise(   sigma_s.x,
											sigma_s.y,
											sigma_s.z,
											sigma_c.x,
											sigma_c.y,
											sigma_c.z,
											scale) );
   
	return;
}

struct add_kinect_noise
{
    float focal_length;
    float theta_1;
    float theta_2;

    float z1;
    float z2;
    float z3;

    add_kinect_noise(float _focal_length,
                 float _theta_1,
                 float _theta_2,
                 float _z1,
                 float _z2,
                 float _z3):
        focal_length(_focal_length),
        theta_1(_theta_1),
        theta_2(_theta_2),
        z1(_z1),
        z2(_z2),
        z3(_z3){}

  __host__ __device__  float3 operator()(const VertexNormalTuple& vertex_normal_tuple,
                                         const unsigned int& thread_id
                                        )
  {
      float3 noisy_3D;
      float3 noisy_lateral = make_float3(0,0,0);
      float3 noisy_axial  = make_float3(0,0,0);

      /// Get the seed up
      clock_t start_time = clock();
      unsigned int seed = hash(thread_id) + start_time;

      thrust::minstd_rand rng(seed);

      const float3 point3D  = thrust::get<0>(vertex_normal_tuple);
      const float3 normal3D = thrust::get<1>(vertex_normal_tuple);

      float depth = point3D.z;
      float my_pi = 22.0f/7.0f;


      /// Subtract the 1 from the dot product; points are represented in homogeneous form with point.w =1
      float dot_prod = normal3D.x*point3D.x +  normal3D.y*point3D.y + normal3D.z*point3D.z ;

      /// xyz of point
      float3 point3D_3 = point3D;
      float norm_point = sqrtf( point3D_3.x* point3D_3.x + point3D_3.y* point3D_3.y + point3D_3.z*point3D_3.z );

      /// negative sign to indicate the position vector of the point starts from the point
      float theta = fabs(acosf(-dot_prod/norm_point));

      float sigma_theta = theta_1 + theta_2*(theta)/(my_pi/2-theta);

      sigma_theta = sigma_theta*(depth)/focal_length;

      thrust::random::normal_distribution<float> normal_noise(0,sigma_theta);
	  float noise_level = normal_noise(rng);
      noisy_lateral.x = point3D.x + noise_level*normal3D.x;
      noisy_lateral.y = point3D.y + noise_level*normal3D.y;
      noisy_lateral.z = point3D.z + noise_level*normal3D.z;

      noisy_3D.x = noisy_lateral.x + noisy_axial.x;
      noisy_3D.y = noisy_lateral.y + noisy_axial.y;
      noisy_3D.z = noisy_lateral.z + noisy_axial.z;

      if ( fabs(my_pi/2 - theta ) <= 8.0/180.0f*my_pi)
      {
          noisy_3D.z = 0.0f;
      }

      return noisy_3D;
  }

};

void launch_add_kinect_noise(float3* points3D,
                             float3* normals3D,
                             float3* noisy_points,
                             const unsigned int stridef3,
                             const unsigned int height,
                             float focal_length,
                             float theta_1,
                             float theta_2,
                             float z1,
                             float z2,
                             float z3)
{
    thrust::device_ptr<float3>points_src(points3D);
    thrust::device_ptr<float3>normals_src(normals3D);
    thrust::device_ptr<float3>points_dest(noisy_points);
    ZipIterator vertex_normal_tuple(thrust::make_tuple(points_src, normals_src));
	try
	{
		thrust::transform( vertex_normal_tuple,
						   vertex_normal_tuple+stridef3*height,
						   thrust::make_counting_iterator(0),
						   points_dest,add_kinect_noise(focal_length,
														theta_1,
														theta_2,
														z1,
														z2,
														z3) );
	}
	catch(thrust::system_error &e)
	{
		// output an error message and exit
		std::cerr << "Error accessing vector element: " << e.what() << std::endl;
		exit(-1);
	}
   
	return;
}


struct colour_from_normals{
    colour_from_normals(){};

    __host__ __device__ float3 operator()(const float3& normal)
    {
        float3 colour;

        colour.x = ( ( normal.x*128.f+128.f ) );
        colour.y = ( ( normal.y*128.f+128.f ) );
        colour.z = ( ( normal.z*128.f+128.f ) );

        return colour;
    }
};

void launch_colour_from_normals(float3* normals,
                                float3* colour,
                                const unsigned int stridef3,
                                const unsigned int height)
{

    thrust::device_ptr<float3> normal_src(normals);
    thrust::device_ptr<float3> colour_dest(colour);
    thrust::transform(normal_src,normal_src + stridef3*height, colour_dest,
                      colour_from_normals());
	return;
}



__global__ void cu_colour_from_normals(const cv::cuda::PtrStepSz<float3> normal, cv::cuda::PtrStepSz<uchar3> color)
{
	const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= normal.cols && nY >= normal.rows) return;

	const float3& nl = normal.ptr(nY)[nX];
	uchar3& colour = color.ptr(nY)[nX];

	float tmp; 
	
	tmp = nl.x*128.f + 128.f;
	tmp = tmp>1.f? 1.f :tmp;
	tmp = tmp<0.f? 0.f :tmp;
	colour.x = uchar( tmp );

	tmp = nl.y*128.f + 128.f;
	tmp = tmp>1.f? 1.f :tmp;
	tmp = tmp<0.f? 0.f :tmp;
    colour.y = uchar( tmp );

	tmp = nl.z*128.f + 128.f;
	tmp = tmp>1.f? 1.f :tmp;
	tmp = tmp<0.f? 0.f :tmp;
    colour.z = uchar( tmp );
	return;
}

void launch_colour_from_normals(const GpuMat& normals, GpuMat* colour)
{
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::cuda::device::divUp(normals.cols, block.x), cv::cuda::device::divUp(normals.rows, block.y));
	//run kernel
	cu_colour_from_normals<<<grid,block>>>( normals,*colour );
	cudaSafeCall ( cudaGetLastError () );
}

struct gaussian_rand{

    float sigma;

    gaussian_rand(float _sigma):sigma(_sigma){};

    __host__ __device__  float2 operator()( float2 point,
                                            const unsigned int& thread_id
                                         )
    {
        float2 noise;

        clock_t start_time = clock();

        unsigned int seed = hash(thread_id) + start_time;

        thrust::minstd_rand rng(seed);
        thrust::random::normal_distribution<float>  randn(0,1);

        noise.x = randn(rng)/sigma;
        noise.y = randn(rng)/sigma;

        return noise;
    }
};

void gaussian_shifts(float2* tex_coods,
                     const unsigned int stridef2,
                     const unsigned int height,
                     const float _sigma)
{

    thrust::device_ptr<float2>coords_src(tex_coods);

    thrust::transform(coords_src,coords_src+stridef2*height,
                      thrust::make_counting_iterator(0),
                      coords_src,
                      gaussian_rand(_sigma) );

}

__global__ void cuda_keneral_add_shift (const PtrStepSz<float> depth, const PtrStepSz<float2> gaussian_shift, PtrStepSz<float> shifted_depth )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= depth.cols || nY >= depth.rows ) return;
	const float2& sh = gaussian_shift.ptr(nY)[nX];
	int2 location = make_int2( round( nX + sh.x ), round( nY + sh.y ) );

	int max;
	max = depth.cols-1;
	location.x = location.x < 0 ? 0 : location.x;
	location.x = location.x > max? max : location.x;
	max = depth.rows-1;
	location.y = location.y < 0 ? 0 : location.y;
	location.y = location.y > max? max : location.y;

	shifted_depth.ptr(nY)[nX] = depth.ptr(location.y)[location.x];
	return;
}

void add_gaussian_shifts( const GpuMat& depth_float_, const GpuMat& gaussian_shift_, GpuMat* depth_shifted_ )
{
	using namespace cv::cuda::device;

	assert( depth_shifted_->cols == depth_float_.cols && depth_shifted_->rows == depth_float_.rows && depth_shifted_->cols == gaussian_shift_.cols && depth_shifted_->rows == gaussian_shift_.rows );

	dim3 block( 8, 8, 1);
	dim3 grid ( divUp (depth_float_.cols, block.x),  divUp (depth_float_.rows, block.y ) );

    cuda_keneral_add_shift<<<grid, block>>>( depth_float_, gaussian_shift_, *depth_shifted_ );
	return;
}


struct gaussian_depth_noise{

    float sigma;

    gaussian_depth_noise(){};

    __host__ __device__  float operator()( float& depth,
                                            const unsigned int& thread_id
                                         )
    {
        float noisy_depth;

        clock_t start_time = clock();

        unsigned int seed = hash(thread_id) + start_time;

        thrust::minstd_rand rng(seed);
        thrust::random::normal_distribution<float>  randn(0,1);

        noisy_depth = (35130/round(35130/round(depth*100) + randn(rng)*(1.0/6.0f) + 0.5))/100;

        return noisy_depth;
    }
};


void add_depth_noise_barronCVPR2013( float* depth_copy,
									 const int stridef1,
									 const int height)
{

    thrust::device_ptr<float>depth_src(depth_copy);

    thrust::transform(depth_src,
                      depth_src+stridef1*height,
                      thrust::make_counting_iterator(0),
                      depth_src,
                      gaussian_depth_noise());
}

__global__ void get_z_coordinate_only(float4* vertex_with_noise,
                                      const unsigned int stridef4,
                                      float* noisy_depth,
                                      const unsigned int stridef1)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    noisy_depth[y*stridef1+x] = vertex_with_noise[y*stridef4+x].z;
}



void launch_get_z_coordinate_only(float4* vertex_with_noise,
                                  const unsigned int stridef4,
                                  const unsigned int width,
                                  const unsigned int height,
                                  float* noisy_depth,
                                  const unsigned int stridef1
                                  )
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    get_z_coordinate_only<<<grid, block>>>(vertex_with_noise,
                                           stridef4,
                                           noisy_depth,
                                           stridef1);
}



__global__ void convert_depth2png (float* noisy_depth,
                                  const unsigned int stridef1,
                                   uint16_t* noisy_depth_png,
                                   const unsigned int strideu16)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    noisy_depth_png[y*strideu16+x] = (unsigned short)(noisy_depth[y*stridef1+x]*5000);
}

void  launch_convert_depth2png(float* noisy_depth,
                               const unsigned int stridef1,
                               unsigned short* noisy_depth_png,
                               const unsigned int strideu16,
                               const unsigned int width,
                               const unsigned int height)
{

    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    convert_depth2png<<<grid, block>>>(noisy_depth,
                                       stridef1,
                                       noisy_depth_png,
                                       strideu16);
}






__device__ float Interpolate(float x0, float x1, float alpha)
{
   return x0 * (1 - alpha) + alpha * x1;
}

__global__ void cu_generateSmoothNoise(float* smoothNoise,
                                  const unsigned int stridef1,
                                  float* baseNoise,
                                  const float samplePeriod,
                                  const float sampleFrequency,
                                  unsigned int width,
                                  unsigned int height)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //calculate the horizontal sampling indices
    int sample_i0 = (x / (int)samplePeriod) * (int)samplePeriod;
    int sample_i1 = (sample_i0 + (int)samplePeriod) % width; //wrap around
    float horizontal_blend = (x - sample_i0) * sampleFrequency;

    //calculate the vertical sampling indices
    int sample_j0 = (y / (int)samplePeriod) * (int)samplePeriod;
    int sample_j1 = (sample_j0 + (int)samplePeriod) % height; //wrap around
    float vertical_blend = (y - sample_j0) * sampleFrequency;

    //blend the top two corners
    float top = Interpolate(baseNoise[sample_i0+stridef1*sample_j0],
                            baseNoise[sample_i1+stridef1*sample_j0],
                            horizontal_blend);

    //blend the bottom two corners
    float bottom = Interpolate(baseNoise[sample_i0+stridef1*sample_j1],
                               baseNoise[sample_i1+stridef1*sample_j1],
                               horizontal_blend);


    smoothNoise[x+y*stridef1] = Interpolate(top, bottom, vertical_blend);

}

void generate_smooth_noise(GpuMat* smoothNoise, //iu::ImageGpu_32f_C1
                           GpuMat* baseNoise, //iu::ImageGpu_32f_C1
                           const float samplePeriod,
                           const float sampleFrequency,
                           const unsigned int width,
                           const unsigned int height)
{

    dim3 blockdim(boost::math::gcd<unsigned>(width, 32), boost::math::gcd<unsigned>(height, 32), 1);
    dim3 griddim( width / blockdim.x, height / blockdim.y);

    cu_generateSmoothNoise<<<griddim,blockdim>>>((float*)smoothNoise->data,
                                           smoothNoise->step,
                                           (float*)baseNoise->data,
                                           samplePeriod,
                                           sampleFrequency,
                                           smoothNoise->cols,
                                           smoothNoise->rows);


}

__global__ void cu_addNoise2Vertex(float4* vertex,
                                   float4* normals,
                                   float4* vertex_with_noise,
                                   const unsigned int stridef4,
                                   float* noise,
                                   const unsigned int stridef1,
                                   const unsigned int width,
                                   const unsigned int height)
{

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if ( x > 0 && x < width && y > 0 && y < height )
    {
        int ind4 = x+y*stridef4;
        int ind1 = x+y*stridef1;

        vertex_with_noise[ind4].x = vertex[ind4].x + noise[ind1]* normals[ind4].x;
        vertex_with_noise[ind4].y = vertex[ind4].y + noise[ind1]* normals[ind4].y;
        vertex_with_noise[ind4].z = vertex[ind4].z + noise[ind1]* normals[ind4].z;
        vertex_with_noise[ind4].w = vertex[ind4].w + 0;
    }

}


void add_noise2vertex(GpuMat* vertex, //iu::ImageGpu_32f_C4
                      GpuMat* normals,//iu::ImageGpu_32f_C4
                      GpuMat* vertex_with_noise,//iu::ImageGpu_32f_C4
                      GpuMat* perlinNoise)//iu::ImageGpu_32f_C1
{

    const int2 imageSize = make_int2(vertex->cols, vertex->rows);
    const int w = imageSize.x;
    const int h = imageSize.y;

    dim3 blockdim(boost::math::gcd<unsigned>(w, 32), boost::math::gcd<unsigned>(h, 32), 1);
    dim3 griddim( w / blockdim.x, h / blockdim.y);

    cu_addNoise2Vertex<<<griddim,blockdim>>>((float4*)vertex->data,
                       (float4 *)normals->data,
                       (float4 *)vertex_with_noise->data,
					   vertex->step,
                       (float*)perlinNoise->data,
                       perlinNoise->step,//stride(),
                       perlinNoise->cols,
                       perlinNoise->rows);

	return;
}


__global__ void cu_verts2depth(
        float* d_depth,
        const float3* d_vert,
        const float2 pp, const float2 fl,
        size_t stridef1, size_t stridef4)
{
    const int x = blockIdx.x*blockDim.x + threadIdx.x;
    const int y = blockIdx.y*blockDim.y + threadIdx.y;
    const int index4 = (x + y*stridef4);

    const float3 v = d_vert[index4];

    if( v.z > 0)// && v.z < 1000)
    {
        float _x_d = ( v.x*fl.x/v.z ) + pp.x;
        float _y_d = ( v.y*fl.y/v.z ) + pp.y;
        int x_d = (int)(_x_d + 0.5f);
        int y_d = (int)(_y_d + 0.5f);
        int index = (x_d + y_d*stridef1);
        d_depth[index] = v.z;
    }
}
//iu::ImageGpu_32f_C1 depth
//iu::ImageGpu_32f_C4 vertex
void convertVerts2Depth(const GpuMat* vertex, GpuMat* depth, float2 pp, float2 fl)
{
    const int2 imageSize = make_int2(depth->cols, depth->rows);
    const size_t stridef1 = depth->cols;
    const size_t stridef4 = vertex->cols;
    const int w = imageSize.x;
    const int h = imageSize.y;

    dim3 blockdim(boost::math::gcd<unsigned>(w, 32), boost::math::gcd<unsigned>(h, 32), 1);
    dim3 griddim( w / blockdim.x, h / blockdim.y);

    cu_verts2depth<<<griddim, blockdim>>>((float*)depth->data,
                                          (float3*)vertex->data,
                                          pp, fl,
                                          stridef1, stridef4);
	cudaSafeCall ( cudaGetLastError () );
	return;
}

__global__ void cuConvertDepth2Verts(   float* depth,
                                        float3* vertex,
                                        const float2 fl,
                                        const float2 pp,
                                        const unsigned int width,
                                        const unsigned int height )
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    vertex[y*width+x] = make_float3( 0.0f,0.0f,0.0f );

    if ( x < width && y < height )
    {
        float depthval = depth[y*width+x];
        vertex[y*width+x] = make_float3( depthval*((float)x-pp.x)/fl.x,
											depthval*((float)y-pp.y)/fl.y,
                                            depthval );
    }
	return;
}

void convertDepth2Verts(const GpuMat& depth, GpuMat* vertex, float2 pp, float2 fl)
{
    const int w = depth.cols;
    const int h = depth.rows;

    dim3 blockdim(boost::math::gcd<unsigned>(w, 32), boost::math::gcd<unsigned>(h, 32), 1);
    dim3 griddim( w / blockdim.x, h / blockdim.y);

    cuConvertDepth2Verts<<<griddim, blockdim>>> ( (float*)depth.data,  (float3*)vertex->data, fl, pp, w, h );
	cudaSafeCall ( cudaGetLastError () );
	return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFastNormalEstimation (const cv::cuda::PtrStepSz<float3> cvgmPts_, cv::cuda::PtrStepSz<float3> cvgmNls_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows ) return;
	float3& fN = cvgmNls_.ptr(nY)[nX];
	if (nX == cvgmPts_.cols - 1 || nY >= cvgmPts_.rows - 1 ){
		fN.x = fN.y = fN.z = 0.f;
		return;
	}
	const float3& pt = cvgmPts_.ptr(nY)[nX];
	const float3& pt1= cvgmPts_.ptr(nY)[nX+1]; //right 
	const float3& pt2= cvgmPts_.ptr(nY+1)[nX]; //down

	if(isnan(pt.z) ||isnan(pt1.z) ||isnan(pt2.z) ){
		fN.x = fN.y = fN.z = 0.f;
		return;
	}//if input or its neighour is NaN,
	float3 v1;
	v1.x = pt1.x-pt.x;
	v1.y = pt1.y-pt.y;
	v1.z = pt1.z-pt.z;
	float3 v2;
	v2.x = pt2.x-pt.x;
	v2.y = pt2.y-pt.y;
	v2.z = pt2.z-pt.z;
	//n = v1 x v2 cross product
	float3 n;
	n.x = v1.y*v2.z - v1.z*v2.y;
	n.y = v1.z*v2.x - v1.x*v2.z;
	n.z = v1.x*v2.y - v1.y*v2.x;
	//normalization
	float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

	if( norm < 1.0e-10 ) {
		fN.x = fN.y = fN.z = 0.f;
		return;
	}//set as NaN,
	n.x /= norm;
	n.y /= norm;
	n.z /= norm;

	if( -n.x*pt.x - n.y*pt.y - n.z*pt.z <0 ){ //this gives (0-pt).dot( n ); 
		fN.x = n.x;
		fN.y = n.y;
		fN.z = n.z;
	}//if facing away from the camera
	else{
		fN.x = -n.x;
		fN.y = -n.y;
		fN.z = -n.z;
	}//else
	return;
}

void cudaFastNormalEstimation(const cv::cuda::GpuMat& cvgmPts_, cv::cuda::GpuMat* pcvgmNls_ )
{
	pcvgmNls_->setTo(0);
	dim3 block (32, 8);
	dim3 grid (cv::cuda::device::divUp (cvgmPts_.cols, block.x), cv::cuda::device::divUp (cvgmPts_.rows, block.y));
	kernelFastNormalEstimation<<<grid, block>>>(cvgmPts_, *pcvgmNls_ );
	cudaSafeCall ( cudaGetLastError () );
}
