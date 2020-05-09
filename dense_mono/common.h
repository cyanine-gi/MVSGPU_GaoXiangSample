#ifndef COMMON_DM_HEADER
#define COMMON_DM_HEADER



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
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积
const double min_cov = 0.1;     // 收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差


// 像素到相机坐标系

inline EigenVector3 px2cam(const EigenVector2 px) {
    return EigenVector3(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
inline EigenVector2 cam2px(const EigenVector3 p_cam) {
    return EigenVector2(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const EigenVector2 &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3_T> &poses,
    cv::Mat &ref_depth
);



bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3_T> &poses,
    cv::Mat &ref_depth) {
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3_T(Quaternionf(data[6], data[3], data[4], data[5]),
                 EigenVector3(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_FLOAT_TYPE);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            FLOAT_T depth = 0;
            fin >> depth;
            ref_depth.ptr<FLOAT_T>(y)[x] = depth / 100.0;
        }

    return true;
}
// 后面这些太简单我就不注释了（其实是因为懒）
using cv::Mat;
using cv::imshow;
using cv::waitKey;
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    imshow("depth_truth", depth_truth * 0.4);
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);
    waitKey(1);
}

void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
    FLOAT_T ave_depth_error = 0;     // 平均误差
    FLOAT_T ave_depth_error_sq = 0;      // 平方误差
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<FLOAT_T>(y)[x] - depth_estimate.ptr<FLOAT_T>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}
#endif
