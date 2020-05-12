#ifndef TYPEDEFS_AND_CONSTS_H
#define TYPEDEFS_AND_CONSTS_H
//typedefs

#define USE_FLOAT_TYPE
#ifdef USE_FLOAT_TYPE
typedef float FLOAT_T;
typedef SE3f SE3_T;
typedef Eigen::Matrix2f EigenMatrix2;
typedef Eigen::Vector3f EigenVector3;
typedef Eigen::Vector2f EigenVector2;
#define CV_FLOAT_TYPE CV_32F
#else
typedef double FLOAT_T;
typedef SE3d SE3_T;
typedef Eigen::Matrix2d EigenMatrix2;
typedef Eigen::Vector3d EigenVector3;
typedef Eigen::Vector2d EigenVector2;
#define CV_FLOAT_TYPE CV_64F
#endif









// parameters
//const int boarder = 20;         // 边缘宽度
//const int width = 640;          // 图像宽度
//const int height = 480;         // 图像高度
//const FLOAT_T fx = 481.2f;       // 相机内参
//const FLOAT_T fy = -480.0f;
//const FLOAT_T cx = 319.5f;
//const FLOAT_T cy = 239.5f;

const int boarder = 20;         // 边缘宽度
const int width = 752;          // 图像宽度
const int height = 480;         // 图像高度
const FLOAT_T fx = 374.f;       // 相机内参
const FLOAT_T fy = -374.0f;
const FLOAT_T cx = 374.0f;
const FLOAT_T cy = 240.0f;


const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差


#endif
