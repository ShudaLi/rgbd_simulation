/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
*/

#ifndef PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_
#define PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_

namespace pcl
{
    namespace device
    {
		template <class T> 
		__device__ __host__ __forceinline__ void swap ( T& a, T& b )
		{
			T c(a); a=b; b=c;
		}
    
		__device__ __host__ __forceinline__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
			return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
		}
		__device__ __host__ __forceinline__ short2 operator - (const short2 s2O1_, const short2 s2O2_){ //can be called from host and device
			return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
		}
		__device__ __host__  __forceinline__ float2 operator * (const float fO1_, const short2 s2O2_){
			return make_float2( fO1_* s2O2_.x, fO1_ * s2O2_.y);
		}
		__device__ __host__ __forceinline__ short2 operator * (const short sO1_, const short2 s2O2_){
			return make_short2( sO1_* s2O2_.x, sO1_ * s2O2_.y);
		}
		__device__ __host__ __forceinline__ float2 operator + (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
			return make_float2(f2O1_.x + f2O2_.x,f2O1_.y + f2O2_.y);
		}
		__device__ __host__ __forceinline__ float2 operator - (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
			return make_float2(f2O1_.x - f2O2_.x,f2O1_.y - f2O2_.y);
		}
		__device__ __host__ __forceinline__ int4 operator + (const int4 n4O1_, const int4 n4O2_){
			return make_int4(n4O1_.x + n4O2_.x, n4O1_.y + n4O2_.y, n4O1_.z+n4O2_.z, n4O1_.w+n4O2_.w);
		}
		__device__ __host__ __forceinline__ int4 operator - (const int4 n4O1_, const int4 n4O2_){
			return make_int4(n4O1_.x - n4O2_.x, n4O1_.y - n4O2_.y, n4O1_.z-n4O2_.z, n4O1_.w-n4O2_.w);
		}
		__device__ __host__ __forceinline__ uchar3 operator * (const uchar3 uc3O1_, const float fO2_){
			return make_uchar3( uchar(uc3O1_.x * fO2_ +.5f), uchar(uc3O1_.y * fO2_+.5f),uchar(uc3O1_.z*fO2_+.5f) );
		}

	
		__device__ __host__ __forceinline__ float3 operator / (const float3& f3O1_, const float& fO2_){
			return make_float3( f3O1_.x / fO2_, f3O1_.y / fO2_, f3O1_.z / fO2_);
		}
		__device__ __host__ __forceinline__ float3 operator * (const float3& f3O1_, const float& fO2_){
			return make_float3( f3O1_.x * fO2_, f3O1_.y * fO2_, f3O1_.z * fO2_);
		}
		__device__ __host__ __forceinline__ float3 operator - (const float3& f3O1_, const float3& f3O2_){
			return make_float3( f3O1_.x - f3O2_.x, f3O1_.y - f3O2_.y, f3O1_.z - f3O2_.z);
		}
		__device__ __host__ __forceinline__ float3 operator + (const float3& f3O1_, const float3& f3O2_){
			return make_float3( f3O1_.x + f3O2_.x, f3O1_.y + f3O2_.y, f3O1_.z + f3O2_.z);
		}
		__device__ __host__ __forceinline__ float3 operator += (const float3& f3O1_, const float& fO2_){
			return make_float3( f3O1_.x + fO2_, f3O1_.y + fO2_, f3O1_.z + fO2_);
		}
		__device__ __host__ __forceinline__ uchar3 operator + (const uchar3 uc3O1_, const uchar3& uc3O2_){
			return make_uchar3( uc3O1_.x + uc3O2_.x, uc3O1_.y + uc3O2_.y, uc3O1_.z + uc3O2_.z);
		}
		__device__ __host__ __forceinline__ float4 operator - (const float4& f3O1_, const float4& f3O2_){
			return make_float4( f3O1_.x - f3O2_.x, f3O1_.y - f3O2_.y, f3O1_.z - f3O2_.z, f3O1_.w - f3O2_.w);
		}
		__device__ __host__ __forceinline__ float4 operator + (const float4& f3O1_, const float4& f3O2_){
			return make_float4( f3O1_.x + f3O2_.x, f3O1_.y + f3O2_.y, f3O1_.z + f3O2_.z, f3O1_.w + f3O2_.w);
		}

		/*__device__  __forceinline__ short2 convert2s2(const float2 f2O1_){ //can be called from host and device
			return make_short2(__float2int_rn(f2O1_.x), __float2int_rn(f2O1_.y));
		}*/



/*
#define PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(type, scalar, op) \
        __device__ __host__ __forceinline__ type & operator op (type & v1, const type & v2) { v1.x op v2.x; v1.y op v2.y; v1.z op v2.z; return v1; } \
        __device__ __host__ __forceinline__ type & operator op (type & v, scalar val)       {  v.x op val;   v.y op val;   v.z op val;  return v;  }

		//PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(float3, float3,-=) 
        PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(float3, float, -=)    
        PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(float3, float, +=)
        PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(float3, float, *=)
		PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(float3, float, /=)


        PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(short3, short, -=) 

        PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(int3, int, +=)

		PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP(uchar3, uchar, +=)

#undef PCL_GPU_IMPLEMENT_COMPOUND_VEC3_OP*/

        __device__ __host__ __forceinline__ float dot(const float3& v1, const float3& v2)
        {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        
        __device__ __host__ __forceinline__ float3 cross(const float3& v1, const float3& v2)
        {
            return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
        }

        ////////////////////////////////
        // four element vectors 


        ////////////////////////////////
        // alltype binary operarators

/*
#define PCL_GPU_IMPLEMENT_VEC_BINOP(type, scalar, op, cop) \
        __device__ __host__ __forceinline__ type operator op (const type & v1, const type & v2) { type r = v1; r cop v2; return r; } \
        __device__ __host__ __forceinline__ type operator op (const type & v1, scalar c)        { type r = v1; r cop c;  return r; }
            
        PCL_GPU_IMPLEMENT_VEC_BINOP(float3, float, -, -=)
        PCL_GPU_IMPLEMENT_VEC_BINOP(float3, float, +, +=)
        PCL_GPU_IMPLEMENT_VEC_BINOP(float3, float, *, *=)
		PCL_GPU_IMPLEMENT_VEC_BINOP(float3, float, /, /=)

        PCL_GPU_IMPLEMENT_VEC_BINOP(short3, short, -, -=)

        PCL_GPU_IMPLEMENT_VEC_BINOP(int3, int, +, +=)

		PCL_GPU_IMPLEMENT_VEC_BINOP(uchar3, uchar, +, +=)

#undef PCL_GPU_IMPLEMENT_VEC_BINOP*/


        ////////////////////////////////
        // tempalted operations vectors 

        template<typename T> __device__ __host__ __forceinline__ float norm(const T& val)
        {
            return sqrtf(dot(val, val));
        }

        template<typename T> __host__ __device__ __forceinline__ float inverse_norm(const T& v)
        {
            return rsqrtf(dot(v, v));
        }

        template<typename T> __host__ __device__ __forceinline__ T normalized(const T& v)
        {
            return v * inverse_norm(v);
        }

		template<typename T> __host__ __device__ __forceinline__ T normalized_safe(const T& v)
        {			
			return (dot(v, v) > 0) ? (v * rsqrtf(dot(v, v))) : v;            
        }
    }//device
}//pcl

#endif /* PCL_GPU_UTILS_DEVICE_VECTOR_MATH_HPP_ */

