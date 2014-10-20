#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "tf/transform_broadcaster.h"
#include "sensor_msgs/Image.h"

#include "KeyframeContainer.h"
#include "KeyframeMatch.h"

#include "pcl_ros/point_cloud.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class MapLocalizer
{
  struct KeyframePositionSorter
  {
    MapLocalizer* ml;
    KeyframePositionSorter(MapLocalizer* ml): ml(ml) {};
    
    bool operator()(KeyframeContainer* kfc1, KeyframeContainer* kfc2)
    {
      return (kfc1->GetTf().block<3,1>(0,3)-ml->currentPosition).norm() < (kfc2->GetTf().block<3,1>(0,3)-ml->currentPosition).norm();
    }
  };


public:
  MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);
  ~MapLocalizer();

private:
  std::vector< KeyframeMatch > FindImageMatches(KeyframeContainer* img, int k, bool usePos = false);
  Eigen::Matrix4f FindImageTf(KeyframeContainer* img, std::vector< KeyframeMatch >, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs);
  void PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf, std::vector< KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs);
  void PublishMap();
  void PublishPointCloud(const std::vector<pcl::PointXYZ>&);
  void PlotTf(Eigen::Matrix4f tf, std::string name);

  void spin(const ros::TimerEvent& e);
  void HandleImage(sensor_msgs::ImageConstPtr msg);
  
  bool WriteDescriptorsToFile(std::string filename);
  bool LoadPhotoscanFile(std::string filename, std::string desc_filename = "", bool load_descs = false);
  Eigen::Matrix4f StringToMatrix4f(std::string);
  std::vector<pcl::PointXYZ> GetPointCloudFromFrames(KeyframeContainer*, KeyframeContainer*);

  std::vector<KeyframeContainer*> keyframes;
  KeyframeContainer* currentKeyframe;
  std::vector<Eigen::Vector3f> positionList;
  Eigen::Vector3f currentPosition;
  bool isLocalized;
  int numLocalizeRetrys;


  ros::NodeHandle nh;
  ros::NodeHandle nh_private;

  ros::Publisher  map_marker_pub;
  ros::Publisher match_marker_pub;
  ros::Publisher tvec_marker_pub;
  ros::Publisher epos_marker_pub;
  ros::Publisher apos_marker_pub;
  ros::Publisher path_marker_pub;
  ros::Publisher pointcloud_pub;
  tf::TransformBroadcaster br;

  ros::Subscriber image_subscriber;

  ros::Timer timer;

  Eigen::Matrix3f K;
  Eigen::VectorXf distcoeff;
};

#endif
