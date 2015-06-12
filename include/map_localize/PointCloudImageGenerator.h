#ifndef _POINTCLOUD_IMAGE_GENERATOR_
#define _POINTCLOUD_IMAGE_GENERATOR_

#include "map_localize/VirtualImageGenerator.h"
#include "pcl_ros/point_cloud.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class PointCloudImageGenerator : public VirtualImageGenerator
{
public:
  PointCloudImageGenerator(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr pc, const Eigen::Matrix3f& K, int rows, int cols);
  virtual cv::Mat GenerateVirtualImage(const Eigen::Matrix4f& pose, cv::Mat& depth, cv::Mat& mask);  
  virtual Eigen::Matrix3f GetK();
  

private:
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr map_cloud;
  Eigen::Matrix3f K;
  int rows;
  int cols;
};
#endif
