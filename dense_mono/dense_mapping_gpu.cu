// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif




#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "opencv2/core/cuda.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"


#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;
using Sophus::SE3f;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"
#include "Timer.h"



using cv::cuda::GpuMat;
using cv::cuda::HostMem;
using cv::Mat;
using cv::cuda::PtrStepSz;



__device__ __forceinline__  EigenVector3 px2cam_gpu(const EigenVector2 px) {
    return EigenVector3(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
__device__ __forceinline__  EigenVector2 cam2px_gpu(const EigenVector3 p_cam) {
    return EigenVector2(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}
__device__ __forceinline__  bool inside_gpu(const EigenVector2 &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

__device__ __forceinline__ FLOAT_T  NCC_gpu(const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr, const EigenVector2 &pt_ref, const EigenVector2 &pt_curr);

__device__ __forceinline__ FLOAT_T getBilinearInterpolatedValue_gpu(const PtrStepSz<uint8_t> &img, const EigenVector2 &pt) {
    //uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    const uint8_t* pixel_ptr = &img(floor(pt(1, 0)),floor(pt(0, 0)));//usage of PtrStepSz.
    const uint8_t* next_ptr = &img(floor(pt(1, 0)+1) , floor(pt(0, 0)) );
    FLOAT_T xx = pt(0, 0) - floor(pt(0, 0));
    FLOAT_T yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * FLOAT_T(pixel_ptr[0]) +
            xx * (1 - yy) * FLOAT_T(pixel_ptr[1]) +
            (1 - xx) * yy * FLOAT_T(next_ptr[0]) +
            xx * yy * FLOAT_T(next_ptr[1])) / 255.0;
}

__device__ bool epipolarSearch_gpu(
    const GpuMat &ref,
    const GpuMat &curr,
    const SE3_T &T_C_R,
    const EigenVector2 &pt_ref,
    const FLOAT_T &depth_mu,
    const FLOAT_T &depth_cov,
    EigenVector2 &pt_curr,
    EigenVector2 &epipolar_direction
);

/**
 * 更新深度滤波器
 * @param pt_ref    参考图像点
 * @param pt_curr   当前图像点
 * @param T_C_R     位姿
 * @param epipolar_direction 极线方向
 * @param depth     深度均值
 * @param depth_cov2    深度方向
 * @return          是否成功
 */



__device__ bool updateDepthFilter_gpu(
    const EigenVector2 &pt_ref,
    const EigenVector2 &pt_curr,
    const SE3_T &T_C_R,
    const EigenVector2 &epipolar_direction,
    PtrStepSz<FLOAT_T> &depth,
    PtrStepSz<FLOAT_T> &depth_cov2
);
__device__ FLOAT_T NCC_gpu(
    const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr,
    const EigenVector2 &pt_ref, const EigenVector2 &pt_curr) {
    // 零均值-归一化互相关
    // 先算均值
    FLOAT_T mean_ref = 0, mean_curr = 0;
    //thrust::device_vector<double> values_ref(2*ncc_window_size+1),values_curr(2*ncc_window_size+1);//vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    const int TotalSize = (2*ncc_window_size+1)*(2*ncc_window_size+1);
    FLOAT_T values_ref[TotalSize];
    FLOAT_T values_curr[TotalSize];
    int index = 0;
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
    {
        for (int y = -ncc_window_size; y <= ncc_window_size; y++)
        {
            uint8_t pixel_val = ref((int)(pt_ref(1, 0)+y),(int)(pt_ref(0, 0)+x));//method to get val directly by PtrStepSz.
            FLOAT_T value_ref = FLOAT_T(pixel_val)/255.0;
            //double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            FLOAT_T value_curr = getBilinearInterpolatedValue_gpu(curr, pt_curr + EigenVector2(x, y));
            mean_curr += value_curr;

            //values_ref.push_back(value_ref);
            //values_curr.push_back(value_curr);
            values_ref[index] = value_ref;
            values_curr[index] = value_curr;
            index++;
        }
    }


    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算 Zero mean NCC
    FLOAT_T numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < TotalSize; i++) {
        FLOAT_T n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}
__device__ bool epipolarSearch_gpu(
    const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr,
    const SE3_T &T_C_R, const EigenVector2 &pt_ref,
    const FLOAT_T &depth_mu, const FLOAT_T &depth_cov,
    EigenVector2 &pt_curr, EigenVector2 &epipolar_direction,PtrStepSz<FLOAT_T> debug_mat) {
    EigenVector3 f_ref = px2cam_gpu(pt_ref);
    f_ref.normalize();
    EigenVector3 P_ref = f_ref * depth_mu;    // 参考帧的 P 向量

    EigenVector2 px_mean_curr = cam2px_gpu(T_C_R * P_ref); // 按深度均值投影的像素
    FLOAT_T d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    EigenVector2 px_min_curr = cam2px_gpu(T_C_R * (f_ref * d_min));    // 按最小深度投影的像素
    EigenVector2 px_max_curr = cam2px_gpu(T_C_R * (f_ref * d_max));    // 按最大深度投影的像素

    EigenVector2 epipolar_line = px_max_curr - px_min_curr;    // 极线（线段形式）
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();
    FLOAT_T half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度
    if (half_length > 100) half_length = 100;   // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    FLOAT_T best_ncc = -1.0;
    EigenVector2 best_px_curr;
    for (FLOAT_T l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        EigenVector2 px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点
        if (!inside_gpu(px_curr))
        {
            continue;
        }
        // 计算待匹配点与参考帧的 NCC
        FLOAT_T ncc = NCC_gpu(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    debug_mat((int)pt_ref(1,0),(int)pt_ref(0,0)) = best_ncc;
    if (best_ncc < 0.85f)      // 只相信 NCC 很高的匹配
    {
        return false;
    }

    pt_curr = best_px_curr;
    return true;
}
__device__ bool updateDepthFilter_gpu(
        const EigenVector2 &pt_ref,
        const EigenVector2 &pt_curr,
        const SE3_T &T_C_R,
        const EigenVector2 &epipolar_direction,
        PtrStepSz<FLOAT_T> &depth,
        PtrStepSz<FLOAT_T> &depth_cov2
)
{
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3_T T_R_C = T_C_R.inverse();
    EigenVector3 f_ref = px2cam_gpu(pt_ref);
    f_ref.normalize();
    EigenVector3 f_curr = px2cam_gpu(pt_curr);
    f_curr.normalize();

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    EigenVector3 t = T_R_C.translation();
    EigenVector3 f2 = T_R_C.so3() * f_curr;
    EigenVector2 b = EigenVector2(t.dot(f_ref), t.dot(f2));
    EigenMatrix2 A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    EigenMatrix2 A_inverse;
    A_inverse(0,0) = A(1,1);
    A_inverse(0,1) = -A(0,1);
    A_inverse(1,0) = -A(1,0);
    A_inverse(1,1) = A(0,0);
    A_inverse*= 1.0/(A(1,1)*A(0,0)-A(0,1)*A(1,0));
    //Vector2d ans = A.inverse() * b; //manually solve equation.
    EigenVector2 ans = A_inverse * b;
    EigenVector3 xm = ans[0] * f_ref;           // ref 侧的结果
    EigenVector3 xn = t + ans[1] * f2;          // cur 结果
    EigenVector3 p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    FLOAT_T depth_estimation = p_esti.norm();   // 深度值

    // 计算不确定性（以一个像素为误差）
    EigenVector3 p = f_ref * depth_estimation;
    EigenVector3 a = p - t;
    FLOAT_T t_norm = t.norm();
    FLOAT_T a_norm = a.norm();
    FLOAT_T alpha = acos(f_ref.dot(t) / t_norm);
    FLOAT_T beta = acos(-a.dot(t) / (a_norm * t_norm));
    EigenVector3 f_curr_prime = px2cam_gpu(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    FLOAT_T beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    FLOAT_T gamma = M_PI - alpha - beta_prime;
    FLOAT_T p_prime = t_norm * sin(beta_prime) / sin(gamma);
    FLOAT_T d_cov = p_prime - depth_estimation;
    FLOAT_T d_cov2 = d_cov * d_cov;

    // 高斯融合
    FLOAT_T mu = depth((int)pt_ref(1, 0),(int)pt_ref(0, 0));
    FLOAT_T sigma2 = depth_cov2((int)pt_ref(1, 0),(int)pt_ref(0, 0));

    FLOAT_T mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    FLOAT_T sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth((int)pt_ref(1, 0),(int)pt_ref(0, 0)) = mu_fuse;
    depth_cov2((int)pt_ref(1, 0),(int)pt_ref(0, 0)) = sigma_fuse2;

    return true;

}
__global__ void update_kernel(PtrStepSz<uint8_t> ref, PtrStepSz<uint8_t> curr, SE3_T T_C_R, PtrStepSz<FLOAT_T> depth, PtrStepSz<FLOAT_T> depth_cov2,
                              PtrStepSz<FLOAT_T> debug_mat)
                              //Here we cannot use any const ref.Just PtrStepSz<Type>.
{

    int y = threadIdx.x+blockIdx.x*blockDim.x;
    int x = threadIdx.y+blockIdx.y*blockDim.y;
//    if(x == 0&&y == 0)
//    {
//        printf("grid_size:%d,%d\n",gridDim.x,gridDim.y);
//    }
    if((x >= boarder&& x < width - boarder) && (y>=boarder&&y<height-boarder))
    {
        // 遍历每个像素
        if (depth_cov2(y,x) < min_cov || depth_cov2(y,x) > max_cov) // 深度已收敛或发散
        {
            //return;
            goto return_to_cpu;
        }
        // 在极线上搜索 (x,y) 的匹配
        EigenVector2 pt_curr;
        EigenVector2 epipolar_direction;
        bool ret = epipolarSearch_gpu(
                    ref,
                    curr,
                    T_C_R,
                    EigenVector2(x, y),
                    depth(y,x),
                    sqrt(depth_cov2(y,x)),
                    pt_curr,
                    epipolar_direction,debug_mat
                    );
        //__syncthreads();
        if (ret == true) // 匹配失败
        {
            // 取消该注释以显示匹配
            //showEpipolarMatch(ref, curr, EigenVector2(x, y), pt_curr);
            //debug_mat(y,x) = 255;
            // 匹配成功，更新深度图
            updateDepthFilter_gpu(EigenVector2(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
        //__syncthreads();

    }
    else
    {
        ;
    }

return_to_cpu:
    __syncthreads();
    return;
}

void initGpuMats(const Mat& ref)
{
//    HostMem ref_gpu_host(HostMem::SHARED),curr_gpu_host(HostMem::SHARED),depth_gpu_host(HostMem::SHARED),depth_cov2_gpu_host(HostMem::SHARED),debug_gpu_host(HostMem::SHARED);
//    ref_gpu_host.create(ref.rows,ref.cols,CV_8U);
//    ref_gpu=ref_gpu_host.createGpuMatHeader();
//    curr_gpu_host.create(ref.rows,ref.cols,CV_8U);
//    curr_gpu=curr_gpu_host.createGpuMatHeader();
//    depth_gpu_host.create(ref.rows,ref.cols,CV_FLOAT_TYPE);
//    depth_gpu=depth_gpu_host.createGpuMatHeader();
//    depth_cov2_gpu_host.create(ref.rows,ref.cols,CV_FLOAT_TYPE);
//    depth_cov2_gpu=depth_cov2_gpu_host.createGpuMatHeader();//make these gpumats static.
//    debug_gpu_host.create(ref.rows,ref.cols,CV_FLOAT_TYPE);
//    debug_mat_gpu=debug_gpu_host.createGpuMatHeader();
}
void update_kernel_wrapper_cpu( Mat& ref,Mat& curr,SE3_T T_C_R,Mat& depth,Mat& depth_cov2)
{
    GpuMat ref_gpu,curr_gpu,depth_gpu,depth_cov2_gpu;//make these gpumats static.
    GpuMat debug_mat_gpu;
    ScopeTimer t_kernel("kernel");
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    cv::Mat debug_mat(ref.rows,ref.cols,CV_FLOAT_TYPE,1);
    cv::cuda::registerPageLocked(ref);
    cv::cuda::registerPageLocked(curr);
    cv::cuda::registerPageLocked(depth);
    cv::cuda::registerPageLocked(depth_cov2);
    cv::cuda::registerPageLocked(debug_mat);
    t_kernel.watch("page_locked");
    ref_gpu.upload(ref);
    curr_gpu.upload(curr);
    depth_gpu.upload(depth);
    depth_cov2_gpu.upload(depth_cov2);
    debug_mat_gpu.upload(debug_mat);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float timeCost_cuda;
    cudaEventElapsedTime(&timeCost_cuda,start,stop);
    cout<<"time to upload:"<<timeCost_cuda<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    t_kernel.watch("uploaded");
    const int MAX_THREAD_SQRT = 16;
    dim3 threads(MAX_THREAD_SQRT, MAX_THREAD_SQRT);
    dim3 grids((ref.rows + MAX_THREAD_SQRT - 1)/MAX_THREAD_SQRT, (ref.cols + MAX_THREAD_SQRT - 1)/ MAX_THREAD_SQRT);
    update_kernel<<<grids, threads>>>(ref_gpu,curr_gpu,T_C_R,depth_gpu,depth_cov2_gpu,debug_mat_gpu);
    t_kernel.watch("kernel func finished");
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCost_cuda,start,stop);
    cout<<"time to call kernel:"<<timeCost_cuda<<endl;
    depth_gpu.download(depth);
    t_kernel.watch("downloaded depth");
    depth_cov2_gpu.download(depth_cov2);
    t_kernel.watch("downloaded depth cov2");
    debug_mat_gpu.download(debug_mat);
    t_kernel.watch("downloaded debug mat");
    cv::imshow("debug_ncc_val",debug_mat);
    cv::waitKey(1);

    cv::cuda::unregisterPageLocked(ref);
    cv::cuda::unregisterPageLocked(curr);
    cv::cuda::unregisterPageLocked(depth);
    cv::cuda::unregisterPageLocked(depth_cov2);
    cv::cuda::unregisterPageLocked(debug_mat);
    t_kernel.watch("unregistered");
//    ref_gpu.release();
//    curr_gpu.release();
//    depth_gpu.release();
//    depth_cov2_gpu.release();
//    debug_mat_gpu.release();
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }
    cudaSetDevice(0);
    cudaProfilerStart();

    // 从数据集读取数据
    vector<string> color_image_files;
    vector<SE3_T> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一张图
    Mat ref = cv::imread(color_image_files[0], 0);                // gray-scale image
    SE3_T pose_ref_TWC = poses_TWC[0];
    FLOAT_T init_depth = 3.0;    // 深度初始值
    FLOAT_T init_cov2 = 3.0;     // 方差初始值
    Mat depth(height, width, CV_FLOAT_TYPE, init_depth);             // 深度图
    Mat depth_cov2(height, width, CV_FLOAT_TYPE, init_cov2);         // 深度图方差


    initGpuMats(depth);

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = cv::imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3_T pose_curr_TWC = poses_TWC[index];
        SE3_T pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R
        update_kernel_wrapper_cpu(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;
    cudaProfilerStop();

    return 0;
}
