/* Copyright (c) 2013 Ankur Handa
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
//eigen
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
using namespace Eigen;
#include <opencv2/opencv.hpp>

#include <opencv2/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
using namespace cv;
#include "VaFRIC.h"

#include <time.h>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>



#define LIVINGROOM

//using namespace boost;

namespace dataset{


Matrix4f vaFRIC::computeTpov_cam(int ref_img_no, int which_blur_sample)
{
    char text_file_name[360];

#ifdef LIVINGROOM
	sprintf(text_file_name,"%s\\scene_%02d_%04d.txt",filebasename.c_str(),which_blur_sample,ref_img_no);
#else
	sprintf(text_file_name,"%s\\scene_%03d.txt",filebasename.c_str(),ref_img_no);
#endif
    
    ifstream cam_pars_file(text_file_name);

    char readlinedata[300];

    Vector3f direction;
    Vector3f upvector;
    Vector3f posvector;


    while(1)
    {
        cam_pars_file.getline(readlinedata,300);

        if ( cam_pars_file.eof())
            break;

        istringstream iss;

        if ( strstr(readlinedata,"cam_dir")!= NULL)
        {
            std::string cam_dir_str(readlinedata);

            cam_dir_str = cam_dir_str.substr(cam_dir_str.find("= [")+3);
            cam_dir_str = cam_dir_str.substr(0,cam_dir_str.find("]"));

            iss.str(cam_dir_str);
            iss >> direction(0);
            iss.ignore(1,',');
            iss >> direction(1);
            iss.ignore(1,',') ;
            iss >> direction(2);
            iss.ignore(1,',');
            //cout << direction.x<< ", "<< direction.y << ", "<< direction.z << endl;

        }

        if ( strstr(readlinedata,"cam_up")!= NULL)
        {

            string cam_up_str(readlinedata);

            cam_up_str = cam_up_str.substr(cam_up_str.find("= [")+3);
            cam_up_str = cam_up_str.substr(0,cam_up_str.find("]"));


            iss.str(cam_up_str);
            iss >> upvector(0);
            iss.ignore(1,',');
            iss >> upvector(1);
            iss.ignore(1,',');
            iss >> upvector(2);
            iss.ignore(1,',');
        }

        if ( strstr(readlinedata,"cam_pos")!= NULL)
        {
            string cam_pos_str(readlinedata);

            cam_pos_str = cam_pos_str.substr(cam_pos_str.find("= [")+3);
            cam_pos_str = cam_pos_str.substr(0,cam_pos_str.find("]"));

            iss.str(cam_pos_str);
            iss >> posvector(0) ;
            iss.ignore(1,',');
            iss >> posvector(1) ;
            iss.ignore(1,',');
            iss >> posvector(2) ;
            iss.ignore(1,',');

        }

    }

    /// z = dir / norm(dir)
    Vector3f z = direction;
    z.normalize();

    /// x = cross(cam_up, z)
    Vector3f x(0,0,0);
	x = upvector.cross(z);
    //x[0] =  upvector.y * z[2] - upvector.z * z[1];
    //x[1] =  upvector.z * z[0] - upvector.x * z[2];
    //x[2] =  upvector.x * z[1] - upvector.y * z[0];

    x.normalize();

    /// y = cross(z,x)
    Vector3f y = z.cross(x);
    //y[0] =  z[1] * x[2] - z[2] * x[1];
    //y[1] =  z[2] * x[0] - z[0] * x[2];
    //y[2] =  z[0] * x[1] - z[1] * x[0];

    Matrix3f R;
	R.col(0) = x;
	R.col(1) = y;
	R.col(2) = z;
	Matrix4f prj; prj.setZero();

	prj.block<3,3>(0,0) = R;
	prj.block<3,1>(0,3) = posvector;
	prj(3,3) = 1.f;
    return prj;
}

void vaFRIC::readDepthFile(int ref_img_no, int which_blur_sample, std::vector<float> &depth_array)
{

    if(!depth_array.size())
        depth_array = std::vector<float>(img_width*img_height,0);

    char depthFileName[300];
#ifdef LIVINGROOM
	sprintf(depthFileName,"%s\\scene_%02d_%04d.depth",filebasename.c_str(),which_blur_sample,ref_img_no);
#else
	sprintf(depthFileName,"%s\\scene_%03d.depth",filebasename.c_str(),ref_img_no);
#endif

    ifstream depthfile;
    depthfile.open(depthFileName);

    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            double val = 0;
            depthfile >> val;
            depth_array[i*img_width+j] = val;
        }
    }

    depthfile.close();
}

void vaFRIC::getEuclidean2PlanarDepth(int ref_img_no, int which_blur_sample, float* depth_array/*, float3* points3D*/)
{
    char depthFileName[300];
	double factor = 1.;
#ifdef LIVINGROOM
    sprintf(depthFileName,"%s\\scene_%02d_%04d.depth",filebasename.c_str(),which_blur_sample,ref_img_no);
#else
	sprintf(depthFileName,"%s\\scene_%03d.depth",filebasename.c_str(),ref_img_no);
	factor = 100.;
#endif

    ifstream depthfile;
    depthfile.open(depthFileName);

    for(int r = 0 ; r < img_height ; r++)
    for (int c = 0 ; c < img_width ; c++)
    {
        double val = 0;
        depthfile >> val;
        val /= factor;

		float u_u0_by_fx = (c-u0)/focal_x;
		float v_v0_by_fy = (r-v0)/focal_y;

		depth_array[ r*img_width+c ] = val / sqrt( u_u0_by_fx*u_u0_by_fx + v_v0_by_fy*v_v0_by_fy + 1 ) ;
    }

    depthfile.close();

/*
    for(int v = 0 ; v < img_height ; v++)
    {
        for(int u = 0 ; u < img_width ; u++)
        {
            float u_u0_by_fx = (u-u0)/focal_x;
            float v_v0_by_fy = (v-v0)/focal_y;

			float z =  depth_array[u+v*img_width] / sqrt( u_u0_by_fx*u_u0_by_fx + v_v0_by_fy*v_v0_by_fy + 1 ) ;
			depth_array[u+v*img_width] = z;

			points3D[u+v*img_width].z = z;
			points3D[u+v*img_width].y = (v_v0_by_fy)*(z);
			points3D[u+v*img_width].x = (u_u0_by_fx)*(z);
        }
    }*/
}

void vaFRIC::convertVerts2Depth(const float3* points3D, float* depth_array )
{

	for(int v = 0 ; v < img_height ; v++)
	{
		for(int u = 0 ; u < img_width ; u++)
		{
			float z = points3D[u+v*img_width].z;
			float y = points3D[u+v*img_width].y;
			float x = points3D[u+v*img_width].x;

			int c = int( focal_x * x / z + u0 + 0.5f );
			int r = int( focal_y * y / z + v0 + 0.5f );

			if( c >=0 && r >=0 && c< img_width && r< img_height ){
				depth_array[u+v*img_width] = z;
			}
		}
	}
}

void vaFRIC::get3Dpositions(const float* depth_array, float3* points3D)
{
	if ( points3D == NULL )
		points3D = new float3[img_width*img_height];


	/// Convert into 3D points
	for(int v = 0 ; v < img_height ; v++)
	{
		for(int u = 0 ; u < img_width ; u++)
		{

			float u_u0_by_fx = (u-u0)/focal_x;
			float v_v0_by_fy = (v-v0)/focal_y;

			float z =  depth_array[u+v*img_width];

			points3D[u+v*img_width].z = z;
			points3D[u+v*img_width].y = (v_v0_by_fy)*(z);
			points3D[u+v*img_width].x = (u_u0_by_fx)*(z);
		}
	}
}


void vaFRIC::get3Dpositions(int ref_img_no, int which_blur_sample, float3* points3D)
{
    if ( points3D == NULL )
        points3D = new float3[img_width*img_height];

    std::vector<float> depth_array(img_width*img_height,0);

    readDepthFile(ref_img_no, which_blur_sample, depth_array);

    /// Convert into 3D points
    for(int v = 0 ; v < img_height ; v++)
    {
        for(int u = 0 ; u < img_width ; u++)
        {

            float u_u0_by_fx = (u-u0)/focal_x;
            float v_v0_by_fy = (v-v0)/focal_y;

            float z =  depth_array[u+v*img_width] / sqrt(u_u0_by_fx*u_u0_by_fx + v_v0_by_fy*v_v0_by_fy + 1 ) ;

//            cout <<" z =" << z << endl;
            points3D[u+v*img_width].z = z;
            points3D[u+v*img_width].y = (v_v0_by_fy)*(z);
            points3D[u+v*img_width].x = (u_u0_by_fx)*(z);
        }
    }
}

void vaFRIC::convertPOV2TUMformat(float *pov_format, unsigned short *tum_format, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            tum_format[i*img_width+j] = (unsigned short)(pov_format[i*img_width+j]*scale_factor);
        }
    }


}



void vaFRIC::convertPOV2TUMformat(float *pov_format, float *tum_format, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            tum_format[i*img_width+j] = (unsigned short)(pov_format[i*img_width+j]*scale_factor);
        }
    }
}


void vaFRIC::convertDepth2NormalisedFloat(float *depth_arrayIn,
                                          float *depth_arrayOut,
                                          float max_depth, float min_depth)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            depth_arrayOut[i*img_width+j] = (depth_arrayIn[i*img_width+j]-min_depth)/(max_depth-min_depth);
        }
    }
}

void vaFRIC::convertDepth2NormalisedFloat(float *depth_arrayIn,
                                          float *depth_arrayOut, int scale_factor)
{
    for(int i = 0 ; i < img_height ; i++)
    {
        for (int j = 0 ; j < img_width ; j++)
        {
            depth_arrayOut[i*img_width+j] = (depth_arrayIn[i*img_width+j]*scale_factor)/65536;
        }
    }
}


void vaFRIC::addDepthNoise(std::vector<float>& depth_arrayIn, std::vector<float>& depth_arrayOut,
                           float z1, float z2,
                           float z3, int ref_img_no,
                           int which_blur_sample )
{
    /// http://www.bnikolic.co.uk/blog/cpp-boost-rand-normal.html

    /// https://github.com/mattdesl/lwjgl-basics/wiki/ShaderLesson6#wiki-GeneratingNormals
    float3* h_points3D = new float3[img_width*img_height];

    get3Dpositions(ref_img_no,which_blur_sample,h_points3D);

    depth_arrayOut.clear();

    if(!depth_arrayOut.size())
        depth_arrayOut = std::vector<float>(img_width*img_height,0);


    for(int i = 0 ; i < img_width; i++ )
    {
        for(int j = 0 ; j < img_height; j++)
        {
            if (i == 0 || j == 0 || i == img_width-1 || j == img_height-1)
                depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width];
            else
            {
                Vector3f vertex_left;
                Vector3f vertex_right;
                Vector3f vertex_up;
                Vector3f vertex_down;

                vertex_left(0)  = h_points3D[i-1+j*img_width].x;
                vertex_left(1)  = h_points3D[i-1+j*img_width].y;
                vertex_left(2)  = h_points3D[i-1+j*img_width].z;

                vertex_right(0) = h_points3D[i+1+j*img_width].x;
                vertex_right(1) = h_points3D[i+1+j*img_width].y;
                vertex_right(2) = h_points3D[i+1+j*img_width].z;

                vertex_up(0)    = h_points3D[i+(j-1)*img_width].x;
                vertex_up(1)    = h_points3D[i+(j-1)*img_width].y;
                vertex_up(2)    = h_points3D[i+(j-1)*img_width].z;

                vertex_down(0)  = h_points3D[i+(j+1)*img_width].x;
                vertex_down(1)  = h_points3D[i+(j+1)*img_width].y;
                vertex_down(2)  = h_points3D[i+(j+1)*img_width].z;

                Vector3f dxv = vertex_right - vertex_left;
                Vector3f dyv = vertex_down  - vertex_up;

                Vector3f normal_vector = dyv.cross( dxv ) ; //dataset::cross(dyv,dxv);

                normal_vector.normalize();

                double c = normal_vector(2);

                double theta = acos(fabs(c));

                double z = depth_arrayIn[i+j*img_width]/100.0;

                double theta_const = (theta/(M_PI/2-theta))*(theta/(M_PI/2-theta))+1E-6;

                double sigma_z = z1 + z2*(z-z3)*(z-z3);// + (z3/sqrt(z)+1E-6)*(theta/(M_PI/2-theta+1E-6))*(theta/(M_PI/2-theta+1E-6));

                sigma_z = sigma_z + (0.0001/sqrt(z))*(theta_const);

                /*boost::variate_generator<boost::mt19937, boost::normal_distribution<> >
                    generator(boost::mt19937(time(0)),
                              boost::normal_distribution<>(0,sigma_z));*/

//                static dataset::RNGType rng;

                static boost::mt19937 rand_number(std::time(0));

//                rng.seed();
                boost::normal_distribution<> rdist(0.0,sigma_z); /**< normal distribution
                                           with mean of 1.0 and standard deviation of 0.5 */

                double noisy_depth = rdist(rand_number)*100;

//                cout << "noisy_depth = " << noisy_depth << endl;

                depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width] + noisy_depth;

                if ( depth_arrayOut[i+j*img_width] <= 0 || fabs(theta-M_PI/2) <= 2*M_PI/180.0f )
                    depth_arrayOut[i+j*img_width] = 0;//depth_arrayIn[i+j*img_width];
                if ( depth_arrayOut[i+j*img_width] >= 5E2 )
                        depth_arrayOut[i+j*img_width] = depth_arrayIn[i+j*img_width];

//                cout << "depth value = " << depth_arrayIn[i+j*img_width] << endl;
//                cout << "rand value generated = " << depth_arrayIn[i+j*img_width] + noisy_depth << endl;


            }
        }
    }

    delete h_points3D;

}

void vaFRIC::addGaussianShift( const Mat& gaussian_shift, const Mat& depth_float, Mat* depth_gaussian ){
	for (int r=0; r< depth_float.rows; r++)
	for (int c=0; c< depth_float.cols; c++)
	{
		const float2& gshift = gaussian_shift.ptr<float2>(r)[c];
		int R = int( r + gshift.y + 0.5f );
		int C = int( c + gshift.x + 0.5f );

		if( R >=0 && C>=0 && R < depth_float.rows && C < depth_float.cols ){
			depth_gaussian->ptr<float>(r)[c] = depth_float.ptr<float>(R)[c];
		}
	}
	return;
}

void vaFRIC::addImageNoise( const Mat& image, const Mat& sigma_s, const Mat& sigma_c, Mat* image_noise){
	for (int r=0; r< image.rows; r++)
		for (int c=0; c< image.cols; c++)
		{
			const float3& s_s = sigma_s.ptr<float3>(r)[c];
			const float3& s_c = sigma_c.ptr<float3>(r)[c];

			const uchar3& pixel = image.ptr<uchar3>(r)[c];

			float3 pf = make_float3( pixel.x/255.f - .5f, pixel.y/255.f - .5f, pixel.z/255.f - .5f );
			float3 pf_n = make_float3( pf.x + s_c.x + s_s.x, pf.y + s_c.y + s_s.y,  pf.z + s_c.x + s_s.z ) ;

			float red = (pf_n.x + .5f) *255.f + .5f; red = red<0.f ? 0.f : red; red = red>255.f ? 255.f : red;
			float g = (pf_n.y + .5f) *255.f + .5f; g = g<0.f ? 0.f : g; g = g>255.f ? 255.f : g;
			float b = (pf_n.z + .5f) *255.f + .5f; b = b<0.f ? 0.f : b; b = b>255.f ? 255.f : b;

			uchar3 pixel_noise = make_uchar3( uchar( red ), uchar( g ), uchar( b ) );
			image_noise->ptr<uchar3>(r)[c] = pixel_noise;
		}
		return;
}


}
