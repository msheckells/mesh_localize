#include "mesh_localize/PointCloudImageGenerator.h"

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

PointCloudImageGenerator::PointCloudImageGenerator(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc, const Eigen::Matrix3f& K, int rows, int cols) :
  map_cloud(pc),
  K(K),
  rows(rows),
  cols(cols)
{
}

Eigen::Matrix3f PointCloudImageGenerator::GetK()
{
  return K;
}

cv::Mat PointCloudImageGenerator::GenerateVirtualImage(const Eigen::Matrix4f& tf, cv::Mat& depths, cv::Mat& mask)
{
  double scale = 1.0;
  int height = rows * scale;
  int width = cols * scale;

  Mat img(height, width, CV_8U, Scalar(0));
  depths = Mat(height, width, CV_32F, Scalar(-1));
  mask = Mat(height, width, CV_8U, Scalar(0));

  Eigen::MatrixXf P(3,4);
  P = K*tf.inverse().block<3,4>(0,0);
  for(unsigned int j = 0; j < map_cloud->points.size(); j++)
  {
    Eigen::Matrix3f Rinv = tf.inverse().block<3,3>(0,0);
    Eigen::Vector3f normal(map_cloud->points[j].normal_x, map_cloud->points[j].normal_y, map_cloud->points[j].normal_z);
    normal = Rinv*normal;

    Eigen::Vector4f hpt(map_cloud->points[j].x, map_cloud->points[j].y, map_cloud->points[j].z, 1);
    Eigen::Vector3f impt = P*hpt;
    impt /= impt(2);
    int dx_idx = floor(impt(0)*scale);
    int dy_idx = floor(impt(1)*scale);
    if(dx_idx < 0  || dx_idx >= width || dy_idx < 0 || dy_idx >= height)
    {
      continue;
    }
    double depth = (tf.inverse()*hpt)(2);
    
    if(depth > 0 /*&& normal(2) < 0*/ && (depths.at<float>(dy_idx, dx_idx) == -1 || depth < depths.at<float>(dy_idx, dx_idx)))
    {
      depths.at<float>(dy_idx, dx_idx) = depth;
      mask.at<uchar>(dy_idx, dx_idx) = 255;
      img.at<uchar>(dy_idx, dx_idx) = (*reinterpret_cast<int*>(&(map_cloud->points[j].rgb)) & 0x0000ff);
    }
  }

  //pyrUp(img, img);//, Size(oldheight, oldwidth));
  medianBlur(img, img, 3);
  medianBlur(depths, depths, 5);
  //medianBlur(mask, mask, 3);
  return img;
}
