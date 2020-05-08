#ifndef COMMON_DM_HEADER
#define COMMON_DM_HEADER

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
inline Vector3d px2cam(const Vector2d px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

// 相机坐标系到像素
inline Vector2d cam2px(const Vector3d p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测一个点是否在图像边框内
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= boarder && pt(1, 0) >= boarder
           && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);



bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
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
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
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
    double ave_depth_error = 0;     // 平均误差
    double ave_depth_error_sq = 0;      // 平方误差
    int cnt_depth_data = 0;
    for (int y = boarder; y < depth_truth.rows - boarder; y++)
        for (int x = boarder; x < depth_truth.cols - boarder; x++) {
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}
#endif
