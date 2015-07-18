#ifndef _MAPLOCALIZER_H_
#define _MAPLOCALIZER_H_

#include <vector>
#include <ros/ros.h>
#include <Eigen/Dense>
#include "tf/transform_broadcaster.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include "MonocularLocalizer.h"
#include "VirtualImageGenerator.h"
#include "KeyframeContainer.h"
#include "KeyframeMatch.h"
#include "MapFeatures.h"
#include "EdgeTrackingUtil.h"
#include "IMUMotionModel.h"

#include "pcl_ros/point_cloud.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
class MapLocalizer
{

  enum LocalizeState
  {
    INIT,
    INIT_PNP,
    LOCAL_INIT,
    PNP,
    EDGES
  } localize_state;

public:

  MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private);
  ~MapLocalizer();

private:
  Eigen::Matrix4f FindImageTfPnp(KeyframeContainer* kcv, const MapFeatures& mf);
  bool FindImageTfVirtualPnp(KeyframeContainer* kcv, Eigen::Matrix4f vimgTf, Eigen::Matrix4f& out, std::string vdesc_type, bool mask_kf, Eigen::Matrix<float, 6, 6>& cov);
  bool FindImageTfVirtualEdges(KeyframeContainer* kcv, Eigen::Matrix4f vimgTf, Eigen::Matrix4f& out, bool mask_kf);
  std::vector<pcl::PointXYZ> GetPointCloudFromFrames(KeyframeContainer*, KeyframeContainer*);
  std::vector<int> FindPlaneInPointCloud(const std::vector<pcl::PointXYZ>& pts);
  Mat GetVirtualImageFromTopic(Mat& depths, Mat& mask);
  Mat GenerateVirtualImage(Eigen::Matrix4f tf, Eigen::Matrix3f K, int height, int width, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, Mat& depth, Mat& mask);

  void UpdateMotionModel(const Eigen::Matrix4f& olfTf, const Eigen::Matrix4f& newTf, 
    const Eigen::Matrix<float, 6, 6>& cov, double dt);
  Eigen::Matrix4f ApplyMotionModel(double dt);
  void ResetMotionModel();

  void PublishPose(Eigen::Matrix4f tf);
  void PublishMap();
  void PublishPointCloud(const std::vector<pcl::PointXYZ>&);
  void PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc);
  void PlotTf(Eigen::Matrix4f tf, std::string name);

  void spin(const ros::TimerEvent& e);
  void HandleImage(const sensor_msgs::ImageConstPtr& msg);
  void HandleVirtualImage(const sensor_msgs::ImageConstPtr& msg);
  void HandleVirtualDepth(const sensor_msgs::ImageConstPtr& msg);
  void UpdateVirtualSensorState(Eigen::Matrix4f tf);

  std::vector<Point3d> PCLToPoint3d(const std::vector<pcl::PointXYZ>& cpvec);
  void ReprojectMask(cv::Mat& dst, const cv::Mat& src, const Eigen::Matrix3f& dstK, 
    const Eigen::Matrix3f& srcK);

  std::vector<CameraContainer*> cameras;
  
  ros::Time img_time_stamp;
  Mat current_image;
  Mat current_virtual_image;
  sensor_msgs::ImageConstPtr current_virtual_depth_msg;

  std::vector<Eigen::Vector3f> positionList;
  Eigen::Matrix4f currentPose;
  bool get_frame;
  bool get_virtual_image;
  bool get_virtual_depth;
  int numPnpRetrys;
  int numLocalizeRetrys;
  double pnpReprojError;
  double pixel_noise;

  MonocularLocalizer* localization_init;
  VirtualImageGenerator* vig;

  ros::Time spin_time;

  MapFeatures map_features;

  std::string ogre_data_dir;
  std::string ogre_cfg_dir;
  std::string ogre_model;
  std::string pc_filename;
  std::string mesh_filename;
  std::string photoscan_filename;
  std::string virtual_image_source;
  std::string pnp_descriptor_type;
  std::string img_match_descriptor_type;
  bool show_pnp_matches;
  bool show_global_matches;
  bool show_debug;
  bool enable_edge_tracking;
  std::string global_localization_alg;
  double image_scale;
  double canny_high_thresh;
  double canny_low_thresh;
  double canny_sigma;
  bool autotune_canny;
  double edge_tracking_dmax;
  int edge_tracking_iterations;
  double pnp_match_radius;
  std::string motion_model;
  bool do_undistort;

  ros::NodeHandle nh;
  ros::NodeHandle nh_private;

  ros::ServiceClient gazebo_client;
  ros::Publisher  estimated_pose_pub;
  ros::Publisher  map_marker_pub;
  ros::Publisher pointcloud_pub;
  tf::TransformBroadcaster br;

  ros::Subscriber image_sub;
  ros::Subscriber virtual_image_sub;
  ros::Subscriber virtual_depth_sub;

  ros::Timer timer;

  Eigen::Matrix4f camera_velocity;
  ros::Time last_spin_time;

  Eigen::Matrix3f K;
  Eigen::Matrix3f K_scaled;
  Eigen::Matrix3f map_K;
  Eigen::Matrix3f virtual_K;
  Eigen::VectorXf distcoeff;
  Eigen::VectorXf map_distcoeff;
  Mat Kcv;
  Mat Kcv_undistort;
  Mat map_Kcv;
  Mat virtual_Kcv;
  Mat distcoeffcv;
  Mat map_distcoeffcv;
  int virtual_height;
  int virtual_width;
  bool init_undistort;
  Mat undistort_map1, undistort_map2;

  IMUMotionModel * imu_mm;
};

#endif
