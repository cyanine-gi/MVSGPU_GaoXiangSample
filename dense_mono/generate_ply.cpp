#include "generate_ply.h"
#include "happly/happly.h"
using cv::Mat;
using cv::imshow;
typedef std::array<double,3> P3d_PLY_T;
inline void getXYZbyUVDepth(//const double& fx,const double& fy,const double& cx,const double& cy,
                const double &depth,
                const double& u,const double& v,
                double& x,double& y,double& z)
{
    Eigen::Vector3d v_((u-cx)/fx,(v-cy)/fy,1);
    v_.normalize();
    v_*=depth;
    x = v_[0];
    y = v_[1];
    z = v_[2];
}

void dump_ply(const Mat& depth_estimate,const Mat& depth_cov2,const Mat& original_img,std::string ply_output_path)
{
    //write a PLY.
    vector<P3d_PLY_T> vVertices;
    vector<array<double,3> > vColors;
    for (int v = boarder; v < original_img.rows-boarder; v++)
    {
        for (int u = boarder; u < original_img.cols-boarder; u++)
        {
            if(depth_cov2.at<FLOAT_T>(v,u) <= min_cov*2)//填充成功的点.
            {
                P3d_PLY_T pt;
                double x_3d,y_3d,z_3d,depth_3d;
                depth_3d = depth_estimate.at<FLOAT_T>(v,u);
                getXYZbyUVDepth(depth_3d,
                                          u,v,
                                          x_3d,y_3d,z_3d);
                if(isnan(x_3d)||isnan(y_3d)||isnan(z_3d) ||
                        isinf(x_3d)||isinf(y_3d)||isinf(z_3d) )
                {
                    continue;
                }
                if(z_3d >10||z_3d<0.01)
                {
                    continue;
                }
                cout<<"uv xyd:"<<u<<","<<v<<","<<x_3d<<"    "<<y_3d<<"  "<<depth_3d<<endl;
                pt[0] = x_3d;pt[1] = y_3d;pt[2] = z_3d;
                vVertices.push_back(pt);
                array<double,3> color;
                color[0] = original_img.at<uint8_t>(v,u)/255.0;
                color[1] = original_img.at<uint8_t>(v,u)/255.0;
                color[2] = original_img.at<uint8_t>(v,u)/255.0;
                vColors.push_back(color);
                //cout<<"uv:"<<u<<","<<v<<endl;
            }
        }
    }
    happly::PLYData plyOut;
    // Add mesh data (elements are created automatically)
    //LOG(INFO)<<"Writing ply with "<<vVertices.size()<<"vertices and "<<vTriangleIDs.size()<<"triangles."<<endl;
    plyOut.addVertexPositions(vVertices);
    plyOut.addVertexColors(vColors);
    vector<vector<int> > vTriangleIDs;
    plyOut.addFaceIndices(vTriangleIDs);

    // Write the object to file
    plyOut.write(ply_output_path, happly::DataFormat::ASCII);
}
//int main()
//{
//    cv::Mat depth_img_ = cv::imread("depth.png");
//    cv::Mat depth_cov2_img_ = cv::imread("depth_cov2.png");
//    cv::Mat original_img = cv::imread("./dataset_output/images/15.png");

//    cv::Mat depth_img,depth_cov2_img;
//    depth_img_.convertTo(depth_img, CV_32F);
//    depth_cov2_img_.convertTo(depth_cov2_img,CV_32F);
//    cout <<"depth_img.type()"<<depth_img.type()<<endl;
//    //assert(depth_img.type()==CV_32F);
//    //assert(depth_cov2_img.type()==CV_32F);
//    dump_ply(depth_img,depth_cov2_img,original_img,"mesh.ply");
//    return 0;
//}
