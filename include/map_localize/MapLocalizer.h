#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "tf/transform_broadcaster.h"
#include "sensor_msgs/Image.h"

#include "KeyframeContainer.h"
#include "KeyframeMatch.h"
#include "MapFeatures.h"

#include "pcl_ros/point_cloud.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
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
  Eigen::Matrix4f FindImageTfSfm(KeyframeContainer* img, std::vector< KeyframeMatch >, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs);
  Eigen::Matrix4f FindImageTfPnp(KeyframeContainer* kcv, const MapFeatures& mf);
  Eigen::Matrix4f FindImageTfVirtualPnp(KeyframeContainer* kcv, Eigen::Matrix4f vimg, Eigen::Matrix3f vimgK);
  std::vector<pcl::PointXYZ> GetPointCloudFromFrames(KeyframeContainer*, KeyframeContainer*);
  std::vector<int> FindPlaneInPointCloud(const std::vector<pcl::PointXYZ>& pts);
  Mat GenerateVirtualImage(Eigen::Matrix4f tf, Eigen::Matrix3f K, int height, int width, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, Mat& depth, Mat& mask);
  Eigen::Matrix4f RansacPnP(std::vector<Point3f> matchPts3d, std::vector<Point2f> matchPts, Eigen::Matrix3f K, Eigen::Matrix4f tfguess);

  
  void PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf);
  void PublishSfmMatchViz(std::vector<KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs);
  void PublishMap();
  void PublishPointCloud(const std::vector<pcl::PointXYZ>&);
  void PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc);
  void PlotTf(Eigen::Matrix4f tf, std::string name);

  void spin(const ros::TimerEvent& e);
  void HandleImage(sensor_msgs::ImageConstPtr msg);
  
  bool WriteDescriptorsToFile(std::string filename);
  bool LoadPhotoscanFile(std::string filename, std::string desc_filename = "", bool load_descs = false);
  Eigen::Matrix4f StringToMatrix4f(std::string);
  std::vector<Point3d> PCLToPoint3d(const std::vector<pcl::PointXYZ>& cpvec);

  std::vector<KeyframeContainer*> keyframes;
  KeyframeContainer* currentKeyframe;
  std::vector<Eigen::Vector3f> positionList;
  Eigen::Vector3f currentPosition;
  bool isLocalized;
  int numLocalizeRetrys;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr map_cloud;
  MapFeatures map_features;

  std::string pc_filename;
  std::string mesh_filename;
  std::string photoscan_filename;

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
  Matx33d Kcv;
  Mat distcoeffcv;
};

#endif
