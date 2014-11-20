#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "tf/transform_broadcaster.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

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
      return (kfc1->GetTf().block<3,1>(0,3)-ml->currentPose.block<3,1>(0,3)).norm() < (kfc2->GetTf().block<3,1>(0,3)-ml->currentPose.block<3,1>(0,3)).norm();
    }
  };

  enum LocalizeState
  {
    INIT,
    INIT_PNP,
    LOCAL_INIT,
    PNP
  } localize_state;

public:

  MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);
  ~MapLocalizer();

private:
  std::vector< KeyframeMatch > FindImageMatches(KeyframeContainer* img, int k, bool usePos = false);
  Eigen::Matrix4f FindImageTfSfm(KeyframeContainer* img, std::vector< KeyframeMatch >, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs);
  Eigen::Matrix4f FindImageTfPnp(KeyframeContainer* kcv, const MapFeatures& mf);
  bool FindImageTfVirtualPnp(KeyframeContainer* kcv, Eigen::Matrix4f vimgTf, Eigen::Matrix3f vimgK, Eigen::Matrix4f& out, std::string vdesc_type);
  std::vector<pcl::PointXYZ> GetPointCloudFromFrames(KeyframeContainer*, KeyframeContainer*);
  std::vector<int> FindPlaneInPointCloud(const std::vector<pcl::PointXYZ>& pts);
  Mat GetVirtualImageFromTopic(Mat& depths, Mat& mask);
  Mat GenerateVirtualImage(Eigen::Matrix4f tf, Eigen::Matrix3f K, int height, int width, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, Mat& depth, Mat& mask);
  bool RansacPnP(const std::vector<Point3f>& matchPts3d, const std::vector<Point2f>& matchPts, Eigen::Matrix3f K, Eigen::Matrix4f tfguess, Eigen::Matrix4f& out);


  void TestFindImageTfSfm();
  
  void PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf);
  void PublishPose(Eigen::Matrix4f tf);
  void PublishSfmMatchViz(std::vector<KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs);
  void PublishMap();
  void PublishPointCloud(const std::vector<pcl::PointXYZ>&);
  void PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc);
  void PlotTf(Eigen::Matrix4f tf, std::string name);

  void spin(const ros::TimerEvent& e);
  void HandleImage(const sensor_msgs::ImageConstPtr& msg);
  void HandleVirtualImage(const sensor_msgs::ImageConstPtr& msg);
  void HandleVirtualDepth(const sensor_msgs::ImageConstPtr& msg);
  void UpdateVirtualSensorState(Eigen::Matrix4f tf);

  bool WriteDescriptorsToFile(std::string filename);
  bool LoadPhotoscanFile(std::string filename, std::string desc_filename = "", bool load_descs = false);
  Eigen::Matrix4f StringToMatrix4f(std::string);
  std::vector<Point3d> PCLToPoint3d(const std::vector<pcl::PointXYZ>& cpvec);

  std::vector<KeyframeContainer*> keyframes;
  KeyframeContainer* currentKeyframe;
  
  Mat current_virtual_image;
  sensor_msgs::ImageConstPtr current_virtual_depth_msg;

  std::vector<Eigen::Vector3f> positionList;
  Eigen::Matrix4f currentPose;
  bool get_virtual_image;
  bool get_virtual_depth;
  int numPnpRetrys;
  int numLocalizeRetrys;

  ros::Time spin_time;

  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr map_cloud;
  MapFeatures map_features;

  std::string pc_filename;
  std::string mesh_filename;
  std::string photoscan_filename;
  std::string virtual_image_source;
  std::string pnp_descriptor_type;
  std::string img_match_descriptor_type;
  bool show_pnp_matches;

  ros::NodeHandle nh;
  ros::NodeHandle nh_private;

  ros::ServiceClient gazebo_client;
  ros::Publisher  estimated_pose_pub;
  ros::Publisher  map_marker_pub;
  ros::Publisher match_marker_pub;
  ros::Publisher tvec_marker_pub;
  ros::Publisher epos_marker_pub;
  ros::Publisher apos_marker_pub;
  ros::Publisher path_marker_pub;
  ros::Publisher pointcloud_pub;
  tf::TransformBroadcaster br;

  ros::Subscriber image_sub;
  ros::Subscriber virtual_image_sub;
  ros::Subscriber virtual_depth_sub;

  ros::Timer timer;

  Eigen::Matrix3f K;
  Eigen::Matrix3f map_K;
  Eigen::Matrix3f virtual_K;
  Eigen::VectorXf distcoeff;
  Eigen::VectorXf map_distcoeff;
  Matx33d Kcv;
  Matx33d map_Kcv;
  Matx33d virtual_Kcv;
  Mat distcoeffcv;
  Mat map_distcoeffcv;
  int virtual_height;
  int virtual_width;
};

#endif
