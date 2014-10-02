#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include "KeyframeContainer.h"
#include "KeyframeMatch.h"


class MapLocalizer
{
public:
  MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);

  std::vector< KeyframeMatch > FindImageMatches(KeyframeContainer* img, int k);
  Eigen::Matrix4f FindImageTf(KeyframeContainer* img, std::vector< KeyframeMatch >);


private:
  bool  LoadPhotoscanFile(std::string filename);
  Eigen::Matrix4f StringToMatrix4f(std::string);


  std::vector<KeyframeContainer*> keyframes;
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;
};

#endif
