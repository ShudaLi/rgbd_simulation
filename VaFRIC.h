#ifndef _vaFRIC_H_
#define _vaFRIC_H_

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


#include<iostream>
#include<string>
//#include<TooN/se3.h>

#include <fstream>
#include <sstream>
#include <stdlib.h>
//#include <TooN/TooN.h>
//#include <TooN/se3.h>
#include <cstring>
//#include <dirent.h>
//#include <cvd/image.h>
//#include <cvd/image_io.h>


#include <vector>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <algorithm>

using namespace std;

namespace dataset{

/*
struct float4{
    float x;
    float y;
    float z;
    float w;
};

struct float3{
	float x;
	float y;
	float z;
};*/

typedef boost::mt19937 RNGType; ///< mersenne twister generator


template<class T>
double gen_normal_3(T &generator)
{
  return generator();
}

// Version that fills a vector
template<class T>
void gen_normal_3(T &generator,
              std::vector<double> &res)
{
  for(size_t i=0; i<res.size(); ++i)
    res[i]=generator();
}


class vaFRIC
{

public:

    vaFRIC(string _filebasename,
           int _img_width,
           int _img_height,
           float _u0,
           float _v0,
           float _focal_x,
           float _focal_y)
        :filebasename(_filebasename),
          txtfilecount(0),
          pngfilecount(0),
          depthfilecount(0),
          img_width(_img_width),
          img_height(_img_height),
          u0(_u0),
          v0(_v0),
          focal_x(_focal_x),
          focal_y(_focal_y)

    {
		//load file names
		boost::filesystem::directory_iterator itrEnd;
		for ( boost::filesystem::directory_iterator itrDir( filebasename );   itrDir != itrEnd;  ++itrDir )	{
			if( itrDir->path().extension() == ".txt" ) {
				txtfilecount++;
			}
			else if( itrDir->path().extension() == ".png" ) {
				pngfilecount++;
			}
			else if( itrDir->path().extension() == ".depth" ) {
				depthfilecount++;
			}
		}

        if ( txtfilecount != pngfilecount || txtfilecount != depthfilecount || pngfilecount != depthfilecount)
            std::cerr<< "Warning: The number of depth files, png files and txt files are not same."<<endl;

    }

    /// Obtain the pose of camera Tpov_cam, with respect to povray world
    Matrix4f computeTpov_cam(int ref_img_no, int which_blur_sample);

    /// POVRay gives euclidean distance of a point from camera, so convert it to obtain depth
    /// If a depth_array pointer is need, use float* depth = &depth_array[0];
	void getEuclidean2PlanarDepth(int ref_img_no, int which_blur_sample, float* depth_array/*, float3* points3D*/);
	/// Wrapper for reading the depth file
    void readDepthFile(int ref_img_no, int which_blur_sample, std::vector<float>& depth_array);

    /// Wrapper for getting the 3D positions
    //void get3Dpositions(int ref_img_no, int which_blur_sample, float4* points3D);    
	void get3Dpositions(int ref_img_no, int which_blur_sample, float3* points3D);
	void get3Dpositions(const float* depth_array, float3* points3D);
    /// Get the number of relevant files
    int getNumberofPoseFiles() { return  txtfilecount ; }
    int getNumberofImageFiles(){ return pngfilecount  ; }
    int getNumberofDepthFiles(){ return depthfilecount; }

    /// Get the png file
    Mat getPNGImage(int ref_img_no, int which_blur_sample)
    {
        char png_file_name[360];
        sprintf(png_file_name,"%s/scene_%02d_%04d.png",filebasename.c_str(),
                which_blur_sample,ref_img_no);

        Mat img;
        img = imread(png_file_name);

        return img;
    }

    /// Convert Depth to TUM format
    void convertPOV2TUMformat(float* pov_format, float* tum_format, int scale_factor);
    void convertPOV2TUMformat(float* pov_format, unsigned short* tum_format, int scale_factor);

    /// Convert Depth to Normalised Float PNG
    void convertDepth2NormalisedFloat(float* depth_arrayIn, float* depth_arrayOut, int scale_factor);

    /// Convert Depth to Normalised Float [min,max] -> [0,1]
    void convertDepth2NormalisedFloat(float *depth_arrayIn,
                                              float *depth_arrayOut,
                                              float max_depth, float min_depth);

    /// Add Noise to the Depth Value
    /// \sigma(z,\theta) = z_{1} + z_{2}(z - z_{3})^2 + (z_{3}/\sqrt{z})*(\theta/(pi/2-theta))^2;
    void addDepthNoise(std::vector<float> &depth_arrayIn, std::vector<float> &depth_arrayOut,
                       float z1, float z2, float z3, int ref_image_no,
                       int which_blur );
	string getDepthFileName( int ref_image_no, int which_blur){
		char depthFileName[300];
		sprintf(depthFileName,"%s\\scene_%02d_%04d.depth.png",filebasename.c_str(),which_blur,ref_image_no);
		string fileName(depthFileName);
		return fileName;
	}


	void convertVerts2Depth(const float3* points3D, float* depth_array );
	void addGaussianShift( const Mat& gaussian_shift, const Mat& depth_float, Mat* depth_gaussian );
	void addImageNoise( const Mat& image, const Mat& sigma_s, const Mat& sigma_c, Mat* image_noise);
private:

    /// Directory variables
    string filebasename;
    int txtfilecount;
    int pngfilecount;
    int depthfilecount;

    /// Camera Intrincs
    int img_width, img_height;
    float u0, v0, focal_x, focal_y;
};

}

#endif
