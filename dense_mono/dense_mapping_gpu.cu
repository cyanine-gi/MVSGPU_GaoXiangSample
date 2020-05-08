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


#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common.h"



using cv::cuda::GpuMat;
using cv::Mat;
using cv::cuda::PtrStepSz;



__device__ inline Vector3d px2cam_gpu(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
__device__ inline Vector2d cam2px_gpu(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}
__device__ inline bool inside_gpu(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

__device__ double  NCC_gpu(const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

__device__ double getBilinearInterpolatedValue_gpu(const PtrStepSz<uint8_t> &img, const Vector2d &pt) {
    //uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    const uint8_t* pixel_ptr = &img(floor(pt(1, 0)),floor(pt(0, 0)));//usage of PtrStepSz.
    const uint8_t* next_ptr = &img(floor(pt(1, 0)+1) , floor(pt(0, 0)) );
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(pixel_ptr[0]) +
            xx * (1 - yy) * double(pixel_ptr[1]) +
            (1 - xx) * yy * double(next_ptr[0]) +
            xx * yy * double(next_ptr[1])) / 255.0;
}

__device__ bool epipolarSearch_gpu(
    const GpuMat &ref,
    const GpuMat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
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
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    PtrStepSz<double> &depth,
    PtrStepSz<double> &depth_cov2
);
__device__ double NCC_gpu(
    const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr,
    const Vector2d &pt_ref, const Vector2d &pt_curr) {
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    //thrust::device_vector<double> values_ref(2*ncc_window_size+1),values_curr(2*ncc_window_size+1);//vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    const int TotalSize = (2*ncc_window_size+1)*(2*ncc_window_size+1);
    double values_ref[TotalSize];
    double values_curr[TotalSize];
    int index = 0;
    for (int x = -ncc_window_size; x <= ncc_window_size; x++)
    {
        for (int y = -ncc_window_size; y <= ncc_window_size; y++)
        {
            uint8_t pixel_val = ref((int)(pt_ref(1, 0)+y),(int)(pt_ref(0, 0)+x));//method to get val directly by PtrStepSz.
            double value_ref = double(pixel_val)/255.0;
            //double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue_gpu(curr, pt_curr + Vector2d(x, y));
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
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < TotalSize; i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}
__device__ bool epipolarSearch_gpu(
    const PtrStepSz<uint8_t> &ref, const PtrStepSz<uint8_t> &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction,PtrStepSz<double> debug_mat) {
    Vector3d f_ref = px2cam_gpu(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;    // 参考帧的 P 向量

    Vector2d px_mean_curr = cam2px_gpu(T_C_R * P_ref); // 按深度均值投影的像素
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px_gpu(T_C_R * (f_ref * d_min));    // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px_gpu(T_C_R * (f_ref * d_max));    // 按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr;    // 极线（线段形式）
    epipolar_direction = epipolar_line;        // 极线方向
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();    // 极线线段的半长度
    if (half_length > 100) half_length = 100;   // 我们不希望搜索太多东西

    // 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;  // 待匹配点
        if (!inside_gpu(px_curr))
        {
            continue;
        }
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC_gpu(ref, curr, pt_ref, px_curr);
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
        const Vector2d &pt_ref,
        const Vector2d &pt_curr,
        const SE3d &T_C_R,
        const Vector2d &epipolar_direction,
        PtrStepSz<double> &depth,
        PtrStepSz<double> &depth_cov2
)
{
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam_gpu(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam_gpu(pt_curr);
    f_curr.normalize();

    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // f2 = R_RC * f_cur
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.so3() * f_curr;
    Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Matrix2d A_inverse;
    A_inverse(0,0) = A(1,1);
    A_inverse(0,1) = -A(0,1);
    A_inverse(1,0) = -A(1,0);
    A_inverse(1,1) = A(0,0);
    A_inverse*= 1.0/(A(1,1)*A(0,0)-A(0,1)*A(1,0));
    //Vector2d ans = A.inverse() * b; //manually solve equation.
    Vector2d ans = A_inverse * b;
    Vector3d xm = ans[0] * f_ref;           // ref 侧的结果
    Vector3d xn = t + ans[1] * f2;          // cur 结果
    Vector3d p_esti = (xm + xn) / 2.0;      // P的位置，取两者的平均
    double depth_estimation = p_esti.norm();   // 深度值

    // 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam_gpu(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    double mu = depth((int)pt_ref(1, 0),(int)pt_ref(0, 0));
    double sigma2 = depth_cov2((int)pt_ref(1, 0),(int)pt_ref(0, 0));

    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth((int)pt_ref(1, 0),(int)pt_ref(0, 0)) = mu_fuse;
    depth_cov2((int)pt_ref(1, 0),(int)pt_ref(0, 0)) = sigma_fuse2;

    return true;

}
__global__ void update_kernel(PtrStepSz<uint8_t> ref, PtrStepSz<uint8_t> curr, SE3d T_C_R, PtrStepSz<double> depth, PtrStepSz<double> depth_cov2,
                              PtrStepSz<double> debug_mat)
                              //Here we cannot use any const ref.Just PtrStepSz<Type>.
{

    int y = threadIdx.x+blockIdx.x*blockDim.x;
    int x = threadIdx.y+blockIdx.y*blockDim.y;
    if(x == 0&&y == 0)
    {
        printf("grid_size:%d,%d\n",gridDim.x,gridDim.y);
    }
    if((x >= boarder&& x < width - boarder) && (y>=boarder&&y<height-boarder))
    {
        // 遍历每个像素
        if (depth_cov2(y,x) < min_cov || depth_cov2(y,x) > max_cov) // 深度已收敛或发散
        {
            return;
        }
        // 在极线上搜索 (x,y) 的匹配
        Vector2d pt_curr;
        Vector2d epipolar_direction;
        bool ret = epipolarSearch_gpu(
                    ref,
                    curr,
                    T_C_R,
                    Vector2d(x, y),
                    depth(y,x),
                    sqrt(depth_cov2(y,x)),
                    pt_curr,
                    epipolar_direction,debug_mat
                    );
        //__syncthreads();
        if (ret == true) // 匹配失败
        {
            // 取消该注释以显示匹配
            //showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);
            //debug_mat(y,x) = 255;
            // 匹配成功，更新深度图
            updateDepthFilter_gpu(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);


        }
        //__syncthreads();

    }
    else
    {
        ;
    }
    __syncthreads();
}
GpuMat ref_gpu,curr_gpu,depth_gpu,depth_cov2_gpu;//make these gpumats static.
GpuMat debug_mat_gpu;
void initGpuMats(const Mat& ref)
{
    ref_gpu.create(ref.rows,ref.cols,CV_8U);
    curr_gpu.create(ref.rows,ref.cols,CV_8U);
    depth_gpu.create(ref.rows,ref.cols,CV_64F);
    depth_cov2_gpu.create(ref.rows,ref.cols,CV_64F);//make these gpumats static.
    debug_mat_gpu.create(ref.rows,ref.cols,CV_64F);
}
void update_kernel_wrapper_cpu(const Mat& ref,const Mat& curr,SE3d T_C_R,Mat& depth,Mat& depth_cov2)
{
    cv::Mat debug_mat(ref.rows,ref.cols,CV_64F,1);
    ref_gpu.upload(ref);
    curr_gpu.upload(curr);
    depth_gpu.upload(depth);
    depth_cov2_gpu.upload(depth_cov2);
    debug_mat_gpu.upload(debug_mat);
    const int MAX_THREAD_SQRT = 16;
    dim3 threads(MAX_THREAD_SQRT, MAX_THREAD_SQRT);
    dim3 grids((ref.rows + MAX_THREAD_SQRT - 1)/MAX_THREAD_SQRT, (ref.cols + MAX_THREAD_SQRT - 1)/ MAX_THREAD_SQRT);
    update_kernel<<<grids, threads>>>(ref_gpu,curr_gpu,T_C_R,depth_gpu,depth_cov2_gpu,debug_mat_gpu);
    depth_gpu.download(depth);
    depth_gpu.download(depth_cov2);
    debug_mat_gpu.download(debug_mat);
    cv::imshow("debug_ncc_val",debug_mat);
    cv::waitKey(1);
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

    // 从数据集读取数据
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一张图
    Mat ref = cv::imread(color_image_files[0], 0);                // gray-scale image
    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;    // 深度初始值
    double init_cov2 = 3.0;     // 方差初始值
    Mat depth(height, width, CV_64F, init_depth);             // 深度图
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // 深度图方差


    initGpuMats(depth);

    for (int index = 1; index < color_image_files.size(); index++) {
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = cv::imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 坐标转换关系： T_C_W * T_W_R = T_C_R
        update_kernel_wrapper_cpu(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth);
        plotDepth(ref_depth, depth);
        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    return 0;
}
