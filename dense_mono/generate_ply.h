#ifndef GENERATE_PLY_H
#define GENERATE_PLY_H
#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
#include <sophus/se3.hpp>
using Sophus::SE3d;
using Sophus::SE3f;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "typedefs_and_consts.h"
using namespace Eigen;
//#include "common.h"
using cv::Mat;
void dump_ply(const Mat& depth_estimate,const Mat& depth_cov2,const Mat& original_img,std::string ply_output_path);
#endif
