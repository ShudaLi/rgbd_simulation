/* Copyright (c) 2013 Ankur Handa and Shuda Li
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

#include <iostream>
#include <stdio.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
using namespace Eigen;
#include <opencv2/opencv.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda/common.hpp>
using namespace cv;

#define INFO
#include "VaFRIC.h"


#include "add_kinect_noise.cuh"

using namespace std;
using namespace cv;

float K[3][3] = { 481.20,   0,  319.50,
					0,  -480.00,  239.50,
					0,        0,    1.00 };

string getImageFileName( const string& path_name_, int ref_image_no, int which_blur){
	char imageFileName[300];
	sprintf(imageFileName,"%s\\scene_%02d_%04d.png",path_name_.c_str(),which_blur,ref_image_no);
	string fileName(imageFileName);
	return fileName;
}
string getDepthFileName( const string& path_name_, int ref_image_no, int which_blur){
	char imageFileName[300];
	sprintf(imageFileName,"%s\\scene_%02d_%04d.depth.png",path_name_.c_str(),which_blur,ref_image_no);
	string fileName(imageFileName);
	return fileName;
}
int init(const string& filebasename){
	int depthfilecount = 0;
	boost::filesystem::directory_iterator itrEnd;
	for ( boost::filesystem::directory_iterator itrDir( filebasename );   itrDir != itrEnd;  ++itrDir )	{
		if( itrDir->path().extension() == ".depth" ) {
			depthfilecount++;
		}
	}
	return depthfilecount;
}

void printNormalImage(const GpuMat& depth_gpu, string& name_){
	GpuMat vertex_gpu; vertex_gpu.create( depth_gpu.rows, depth_gpu.cols,  CV_32FC3 );
	GpuMat normal_gpu; normal_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_32FC3 );
	GpuMat normal_image_float_gpu; normal_image_float_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_32FC3 );
	GpuMat normal_image_gpu; normal_image_gpu.create( depth_gpu.rows, depth_gpu.cols, CV_8UC3 );
	////for debug
	convertDepth2Verts( depth_gpu, &vertex_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
	cudaFastNormalEstimation( vertex_gpu, &normal_gpu );
	launch_colour_from_normals( (float3*)normal_gpu.data , (float3*)normal_image_float_gpu.data, normal_gpu.cols, normal_gpu.rows );

	normal_image_float_gpu.convertTo(normal_image_gpu,CV_8UC3);
	Mat normal_image; normal_image_gpu.download(normal_image);
	imwrite(name_.c_str(),normal_image);
}

int main(int argc, char** argv)
{
	string input_path_name ("D:\\Downloads\\icl\\noise_lvl_0\\living_room_traj0_loop\\");
	string output_path_name("D:\\Downloads\\icl\\noise_lvl_1\\living_room_traj0_loop\\");
	const int nTotal = init(input_path_name);
    cout<<"Number of text files = " << nTotal << endl;

	Mat depth_float; depth_float.create(480,640,CV_32FC1);
	Mat depth; depth.create(480,640,CV_16UC1);
	Mat vertex; vertex.create(480,640,CV_32FC3);
	Mat normal_dist; normal_dist.create(480,640,CV_32FC1);

	GpuMat vertex_gpu; vertex_gpu.create(480,640,CV_32FC3);
	GpuMat normal_gpu; normal_gpu.create(480,640,CV_32FC3);
	GpuMat noisy_vertex_gpu; noisy_vertex_gpu.create(480,640,CV_32FC3);
	GpuMat depth_gpu; depth_gpu.create(480,640,CV_16UC1);
	GpuMat depth_float_gpu; depth_float_gpu.create(480,640,CV_32FC1);
	GpuMat depth_shifted_gpu; depth_shifted_gpu.create(480,640,CV_32FC1);
	GpuMat gaussian_2d_shift_gpu; gaussian_2d_shift_gpu.create(480,640,CV_32FC2);
	GpuMat normal_image_gpu_float; normal_image_gpu_float.create(480,640,CV_32FC3);
	GpuMat normal_image_gpu; normal_image_gpu.create(480,640,CV_8UC3);

	float3 sigma_c = make_float3(0.0045f, 0.0038f, 0.005f);
	float3 sigma_s = make_float3(0.0104f, 0.0066f, 0.0106f);

	Mat noisy_image; noisy_image.create( 480, 640, CV_8UC3 );
	Mat normal_image; 
	
	for (int img_no=0; img_no< nTotal; img_no++)
	{
		// add image noise
		{
			string imgName = getImageFileName( input_path_name, img_no, 0 );

			Mat image = imread(imgName.c_str());
			GpuMat gpu_image; gpu_image.upload(image);
			GpuMat gpu_image_float; gpu_image.convertTo(gpu_image_float,CV_32FC3);
			GpuMat gpu_noisy_image_float = gpu_image_float.clone();

			launch_add_camera_noise((float3*)gpu_image_float.data, (float3*)gpu_noisy_image_float.data, sigma_s, sigma_c, gpu_image_float.cols, gpu_image_float.rows, 255.f );

			GpuMat gpu_noisy_image;
			gpu_noisy_image_float.convertTo( gpu_noisy_image, CV_8UC3, 255.f );
			gpu_noisy_image.download( noisy_image );
			string out_imgName = getImageFileName( output_path_name, img_no, 0 );
			imwrite( out_imgName.c_str(), noisy_image );
		}

		//add depth noise
		//dataset.getEuclidean2PlanarDepth( img_no, 0, (float*)depth_float.data );

		string depName = getDepthFileName( input_path_name, img_no, 0 );
		depth = imread( depName.c_str(), IMREAD_UNCHANGED );
		depth_gpu.upload( depth );
		depth_gpu.convertTo( depth_float_gpu, CV_32FC1, 1/1000.f );
		//printNormalImage(depth_float_gpu,string("ni_1.png"));

		//1.
		convertDepth2Verts( depth_float_gpu, &vertex_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
		cudaFastNormalEstimation( vertex_gpu, &normal_gpu );
		launch_add_kinect_noise( (float3*)vertex_gpu.data,
								 (float3*)normal_gpu.data,
								 (float3*)noisy_vertex_gpu.data,
								 vertex_gpu.cols,
								 vertex_gpu.rows,
								 480, 0.8, 0.035,
								 0, 0, 0 );
		//convert vertex to depth
		convertVerts2Depth( &noisy_vertex_gpu, &depth_float_gpu, make_float2(K[0][2],K[1][2]), make_float2(K[0][0],K[1][1]) );
		//printNormalImage(depth_float_gpu,string("ni_2.png"));

		//2. 
		gaussian_shifts( (float2*)gaussian_2d_shift_gpu.data, gaussian_2d_shift_gpu.cols, gaussian_2d_shift_gpu.rows, 3.f );
		add_gaussian_shifts( depth_float_gpu, gaussian_2d_shift_gpu, &depth_shifted_gpu );
		//printNormalImage( depth_shifted_gpu, string("ni_3.png") );
		
		//3.
		add_depth_noise_barronCVPR2013( (float*)depth_shifted_gpu.data, depth_shifted_gpu.cols, depth_shifted_gpu.rows );
		//printNormalImage( depth_shifted_gpu, string("ni_4.png") );

		//convert depth from metre to 1000 mm 
		depth_shifted_gpu.convertTo( depth_gpu, CV_16UC1, 1000 );
		depth_gpu.download(depth);

		string out_depName = getDepthFileName( output_path_name, img_no, 0 );
		imwrite( out_depName.c_str(), depth );
		cout << img_no << " ";
	}

	return 1;
}
