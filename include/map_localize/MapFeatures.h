#ifndef _MAPFEATURES_H_
#define _MAPFEATURES_H_

#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_types.h>

#include "map_localize/KeyframeContainer.h"

using namespace cv;

class MapFeatures
{
public:
  MapFeatures(std::vector<KeyframeContainer*>& kcv,  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
  MapFeatures(){};
  
  Mat GetDescriptors() const;
  std::vector<pcl::PointXYZ> GetKeypoints() const;
private:
  std::vector<KeyframeContainer*> kcv;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  std::vector<pcl::PointXYZ> keypoints;
  Mat descriptors;
};

#endif
