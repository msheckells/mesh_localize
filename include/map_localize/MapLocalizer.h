#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "tf/transform_broadcaster.h"
#include "sensor_msgs/Image.h"

#include "KeyframeContainer.h"
#include "KeyframeMatch.h"


class MapLocalizer
{
public:
  MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);
  ~MapLocalizer();

  std::vector< KeyframeMatch > FindImageMatches(KeyframeContainer* img, int k);
  Eigen::Matrix4f FindImageTf(KeyframeContainer* img, std::vector< KeyframeMatch >, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs);
  void PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf, std::vector< KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs);

  void spin(const ros::TimerEvent& e);
  void HandleImage(sensor_msgs::Image msg);

private:
  bool WriteDescriptorsToFile(std::string filename);
  bool LoadPhotoscanFile(std::string filename, std::string desc_filename = "", bool load_descs = false);
  Eigen::Matrix4f StringToMatrix4f(std::string);


  std::vector<KeyframeContainer*> keyframes;
  ros::NodeHandle nh;
  ros::NodeHandle nh_private;

  ros::Publisher  map_marker_pub;
  ros::Publisher match_marker_pub;
  ros::Publisher tvec_marker_pub;
  ros::Publisher epos_marker_pub;
  ros::Publisher apos_marker_pub;
  tf::TransformBroadcaster br;

  ros::Subscriber image_subscriber;

  ros::Timer timer;
};

#endif
