#include "map_localize/MapLocalizer.h"
#include <algorithm>
#include <sstream>
#include <cstdlib>     
#include <time.h> 
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <cv_bridge/cv_bridge.h>

#include <tf/transform_broadcaster.h>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "map_localize/OgreImageGenerator.h"
#include "map_localize/FindCameraMatrices.h"
#include "map_localize/Triangulation.h"
#include "map_localize/ASiftDetector.h"
#include "map_localize/ImageDbUtil.h"
#include "map_localize/PnPUtil.h"
#include "map_localize/EdgeTrackingUtil.h"
#include "map_localize/PointCloudImageGenerator.h"
#include "map_localize/FeatureMatchLocalizer.h"
#include "map_localize/FABMAPLocalizer.h"
#include "map_localize/DepthFeatureMatchLocalizer.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "gazebo_msgs/LinkState.h"
#include "gazebo_msgs/SetLinkState.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>

#include <gcop/so3.h>

//#include <kvld/kvld.h>

MapLocalizer::MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private):
    init_undistort(true),
    get_frame(true),
    get_virtual_image(false),
    get_virtual_depth(false),
    numPnpRetrys(0),
    numLocalizeRetrys(0),
    nh(nh),
    nh_private(nh_private)
{
  // Get Params
  bool load_descriptors;
  std::string descriptor_filename;

  if (!nh_private.getParam ("point_cloud_filename", pc_filename))
    pc_filename = "bin/map_points.pcd";
  if (!nh_private.getParam ("mesh_filename", mesh_filename))
    mesh_filename = "bin/map.stl";
  if (!nh_private.getParam ("photoscan_filename", photoscan_filename))
    photoscan_filename = "/home/matt/Documents/campus_doc.xml";
  if (!nh_private.getParam ("load_descriptors", load_descriptors))
    load_descriptors = false;
  if (!nh_private.getParam ("show_pnp_matches", show_pnp_matches))
    show_pnp_matches = false;
  if (!nh_private.getParam ("show_debug", show_debug))
    show_debug = false;
  if (!nh_private.getParam ("show_global_matches", show_global_matches))
    show_global_matches = false;
  if (!nh_private.getParam("descriptor_filename", descriptor_filename))
    descriptor_filename = "";
  if (!nh_private.getParam("ogre_data_dir", ogre_data_dir))
    ogre_data_dir = "";
  if (!nh_private.getParam("ogre_cfg_dir", ogre_cfg_dir))
    ogre_cfg_dir = "";
  if (!nh_private.getParam("ogre_model", ogre_model))
    ogre_model = "";
  if(!nh_private.getParam("virtual_image_source", virtual_image_source))
    virtual_image_source = "point_cloud";
  if(!nh_private.getParam("pnp_descriptor_type", pnp_descriptor_type))
    pnp_descriptor_type = "orb";
  if(!nh_private.getParam("img_match_descriptor_type", img_match_descriptor_type))
    img_match_descriptor_type = "asurf";
  if(!nh_private.getParam("global_localization_alg", global_localization_alg))
    global_localization_alg = "feature_match";
  if(!nh_private.getParam("image_scale", image_scale))
    image_scale = 1.0;
  if(!nh_private.getParam("canny_sigma", canny_sigma))
    canny_sigma = 0.33;
  if(!nh_private.getParam("canny_high_thresh", canny_high_thresh))
    canny_high_thresh = 200;
  if(!nh_private.getParam("canny_low_thresh", canny_low_thresh))
    canny_low_thresh = 80;
  if(!nh_private.getParam("edge_tracking_dmax", edge_tracking_dmax))
    edge_tracking_dmax = 15;
  if(!nh_private.getParam("autotune_canny", autotune_canny))
    autotune_canny = false;
  if(!nh_private.getParam("enable_edge_tracking", enable_edge_tracking))
    enable_edge_tracking = false;
  if(!nh_private.getParam("edge_tracking_iterations", edge_tracking_iterations))
    edge_tracking_iterations = 1;
  if(!nh_private.getParam("pnp_match_radius", pnp_match_radius))
    pnp_match_radius = -1;
  if(!nh_private.getParam("motion_model", motion_model))
    motion_model = "CONSTANT";
  if(!nh_private.getParam("do_undistort", do_undistort))
    do_undistort = true;
  if(!nh_private.getParam("pixel_noise", pixel_noise))
    pixel_noise = 3;
  
  if(enable_edge_tracking)
  {
    EdgeTrackingUtil::show_debug = show_debug;
    EdgeTrackingUtil::canny_high_thresh = canny_high_thresh;
    EdgeTrackingUtil::canny_low_thresh = canny_low_thresh;
    EdgeTrackingUtil::canny_sigma = canny_sigma;
    EdgeTrackingUtil::autotune_canny = autotune_canny;
    EdgeTrackingUtil::dmax = edge_tracking_dmax;
  }

  //TODO: read from param file.  Hard-coded, based on DSLR
  map_K << 1799.352269, 0, 1799.029749, 0, 1261.4382272, 957.3402899, 0, 0, 1;
  map_distcoeff = Eigen::VectorXf(5);
  map_distcoeff << 0, 0, 0, 0, 0;
  //map_distcoeff << -.0066106, .04618129, -.00042169, -.004390247, -.048470351;
  map_Kcv = (Mat_<double>(3,3) << map_K(0,0), map_K(0,1), map_K(0,2),
              map_K(1,0), map_K(1,1), map_K(1,2),
              map_K(2,0), map_K(2,1), map_K(2,2)); 
  map_distcoeffcv = (Mat_<double>(5,1) << map_distcoeff(0), map_distcoeff(1), map_distcoeff(2), map_distcoeff(3), map_distcoeff(4)); 

  if(image_scale != 1.0)
  {
    ROS_INFO("Scaling images by %f", image_scale);
  }

  ROS_INFO("Using %s for pnp descriptors and %s for image matching descriptors", pnp_descriptor_type.c_str(), img_match_descriptor_type.c_str());

  if(global_localization_alg == "feature_match")
  {
    vector<CameraContainer*> image_db;
    if(!ImageDbUtil::LoadPhotoscanFile(photoscan_filename, image_db, map_Kcv, map_distcoeffcv))
    {
      return;
    }
    ROS_INFO("Using Photoscan object feature matching for initialization");
    localization_init = new FeatureMatchLocalizer(image_db, img_match_descriptor_type, show_global_matches, load_descriptors, descriptor_filename);
  }
  else if(global_localization_alg == "depth_feature_match")
  {
    vector<KeyframeContainer*> image_db;
    if(!ImageDbUtil::LoadOgreDataDir(ogre_data_dir, image_db))
    {
      return;
    }
    ROS_INFO("Using Ogre object feature matching for initialization");
    if(img_match_descriptor_type != "surf")
    {
      ROS_ERROR("img_match_descriptor_type must be 'surf' when using OGRE ImageDb");
      return;
    }
    localization_init = new DepthFeatureMatchLocalizer(image_db, img_match_descriptor_type, show_global_matches);
  }
  else if(global_localization_alg == "fabmap")
  {
    vector<CameraContainer*> image_db;
    if(!ImageDbUtil::LoadPhotoscanFile(photoscan_filename, image_db, map_Kcv, map_distcoeffcv))
    {
      return;
    }
    ROS_INFO("Using OpenFABMAP for initialization");
    if(img_match_descriptor_type != "surf")
    {
      ROS_ERROR("img_match_descriptor_type must be 'surf' when using OpenFABMAP");
      return;
    }
    localization_init = new FABMAPLocalizer(image_db, img_match_descriptor_type, show_global_matches, load_descriptors, descriptor_filename);
  }
  else
  {
    ROS_ERROR("%s is not a valid initialization option", global_localization_alg.c_str());
    return; 
  }

  std::srand(time(NULL));

  ROS_INFO("Waiting for camera_info...");
  sensor_msgs::CameraInfoConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("camera_info", nh);
  ROS_INFO("camera_info received");

  K << msg->K[0], msg->K[1], msg->K[2],
                 msg->K[3], msg->K[4], msg->K[5],
                 msg->K[6], msg->K[7], msg->K[8];
  K_scaled = image_scale * K;
  K_scaled(2,2) = 1;
  distcoeff = Eigen::VectorXf(5);
  distcoeff << msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4];
  distcoeffcv = (Mat_<double>(5,1) << distcoeff(0), distcoeff(1), distcoeff(2), distcoeff(3), distcoeff(4)); 
  Kcv_undistort = (Mat_<double>(3,3) << K(0,0), K(0,1), K(0,2),
                                       K(1,0), K(1,1), K(1,2),
                                       K(2,0), K(2,1), K(2,2)); 
  Kcv = image_scale*Kcv_undistort;
  Kcv.at<double>(2,2) = 1;
 
  estimated_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/map_localize/estimated_pose", 1);
  map_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/map", 1);
  pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/map_localize/pointcloud", 1);

  image_sub = nh.subscribe<sensor_msgs::Image>("image", 1, &MapLocalizer::HandleImage, this, ros::TransportHints().tcpNoDelay());

  if(virtual_image_source == "point_cloud")
  {
    ROS_INFO("Using PCL point cloud for virtual image generation");
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr map_cloud = 
      pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ROS_INFO("Loading point cloud %s", pc_filename.c_str());
    if(pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (pc_filename, *map_cloud) == -1)
    {
      std::cout << "Could not open point cloud " << pc_filename << std::endl;
      return;
    }
    ROS_INFO("Successfully loaded point cloud");
    vig = new PointCloudImageGenerator(map_cloud, K, msg->height, msg->width); 
  }
  else if(virtual_image_source == "ogre")
  {
    ROS_INFO("Using Ogre for virtual image generation");
    vig = new OgreImageGenerator(ogre_cfg_dir, ogre_model);
  }
  else if(virtual_image_source == "gazebo")
  {
    ROS_INFO("Using Gazebo for virtual image generation");
    gazebo_client = nh.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");
    virtual_image_sub = nh.subscribe<sensor_msgs::Image>("virtual_image", 1, &MapLocalizer::HandleVirtualImage, this, ros::TransportHints().tcpNoDelay());
    virtual_depth_sub = nh.subscribe<sensor_msgs::Image>("virtual_depth", 1, &MapLocalizer::HandleVirtualDepth, this, ros::TransportHints().tcpNoDelay());

    ROS_INFO("Waiting for camera calibration info...");
  
    sensor_msgs::CameraInfoConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("virtual_caminfo", nh);
    virtual_K << msg->K[0], msg->K[1], msg->K[2],
                 msg->K[3], msg->K[4], msg->K[5],
                 msg->K[6], msg->K[7], msg->K[8];
    virtual_Kcv = (Mat_<double>(3,3) << virtual_K(0,0), virtual_K(0,1), virtual_K(0,2),
                                        virtual_K(1,0), virtual_K(1,1), virtual_K(1,2),
                                        virtual_K(2,0), virtual_K(2,1), virtual_K(2,2)); 
    virtual_width = msg->width;
    virtual_height = msg->height;
  
    ROS_INFO("Calibration info received");
  }
  else
  {
    ROS_ERROR("%s is not a valid virtual image source", virtual_image_source.c_str());
    return;
  }

  if(motion_model == "IMU")
  {
    imu_mm = new IMUMotionModel();
    ROS_INFO("Calibrating gyros...");
    while(!imu_mm->isCalibrated())
    {
    }
    ROS_INFO("Done calibrating gyros...");
  }
  else
  {
    imu_mm = NULL;
  }

  if(show_pnp_matches)
  { 
    namedWindow( "PnP Matches", WINDOW_NORMAL );
    namedWindow( "PnP Match Inliers", WINDOW_NORMAL );
  }

  ResetMotionModel();
  localize_state = INIT;

  spin_time = ros::Time::now();
  timer = nh_private.createTimer(ros::Duration(0.001), &MapLocalizer::spin, this);
}

MapLocalizer::~MapLocalizer()
{
  if(imu_mm)
    delete imu_mm;
}

void MapLocalizer::HandleImage(const sensor_msgs::ImageConstPtr& msg)
{
  if(get_frame)
  {
    ROS_INFO("Processing new image");
    img_time_stamp = ros::Time::now(); 
    if((localize_state == PNP || localize_state == INIT_PNP) && virtual_image_source == "gazebo")
    {
      get_virtual_depth = true;
      get_virtual_image = true;
    }

    ros::Time start = ros::Time::now();
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(msg);
    Mat img_undistort;
    Mat image = cvImg->image;
    //undistort(cvImg->image, current_image, Kcv_undistort, distcoeffcv);
    if(image_scale != 1.0)
    {
      resize(image, image, Size(0,0), image_scale, image_scale);
    }
    if(do_undistort)
    {
      if(init_undistort)
      {
        initUndistortRectifyMap(Kcv, distcoeffcv, Mat::eye(3, 3, CV_64F), Kcv, 
          Size(image.cols, image.rows), CV_32FC1, undistort_map1, undistort_map2);
        init_undistort = false;
      }
      remap(image, current_image, undistort_map1, undistort_map2, INTER_LINEAR);
      //undistort(image, current_image, Kcv, distcoeffcv);
    }
    else
    {
      current_image = image;
    }
    ROS_INFO("Image process time: %f", (ros::Time::now()-start).toSec());  
    get_frame = false;
  }
}

void MapLocalizer::HandleVirtualImage(const sensor_msgs::ImageConstPtr& msg)
{
  if(get_virtual_image)
  {
    ROS_INFO("Got virtual image");
    current_virtual_image = cv_bridge::toCvCopy(msg)->image;
    get_virtual_image = false;
  }
}

void MapLocalizer::HandleVirtualDepth(const sensor_msgs::ImageConstPtr& msg)
{
  if(get_virtual_depth)
  {
    ROS_INFO("Got virtual depth");
    current_virtual_depth_msg = msg;
    get_virtual_depth = false;
  }
}

void MapLocalizer::UpdateVirtualSensorState(Eigen::Matrix4f tf)
{
  if(virtual_image_source == "gazebo")
  {
    gazebo_msgs::SetLinkState vimg_state_srv;
    gazebo_msgs::LinkState vimg_state_msg;
    vimg_state_msg.link_name = "kinect::link";

    vimg_state_msg.pose.position.x = tf(0,3);
    vimg_state_msg.pose.position.y = tf(1,3);
    vimg_state_msg.pose.position.z = tf(2,3);

    Eigen::Matrix3f rot = tf.block<3,3>(0,0)*Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitY())*Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitX());
    Eigen::Quaternionf q(rot);
    q.normalize();
    vimg_state_msg.pose.orientation.x = q.x();
    vimg_state_msg.pose.orientation.y = q.y();
    vimg_state_msg.pose.orientation.z = q.z();
    vimg_state_msg.pose.orientation.w = q.w();
  
    vimg_state_msg.twist.linear.x = 0;
    vimg_state_msg.twist.linear.y = 0;
    vimg_state_msg.twist.linear.z = 0;
    vimg_state_msg.twist.angular.x = 0;
    vimg_state_msg.twist.angular.y = 0;
    vimg_state_msg.twist.angular.z = 0;

    vimg_state_srv.request.link_state = vimg_state_msg;

    ros::Time start = ros::Time::now();
    if(!gazebo_client.call(vimg_state_srv))
    {
      ROS_ERROR("Failed to contact gazebo set_link_state service");
    }
    ROS_INFO("set_link_state time: %f", (ros::Time::now()-start).toSec());
    usleep(1e4);
  }
}

void MapLocalizer::PublishPose(Eigen::Matrix4f tf)
{
  geometry_msgs::PoseStamped pose;

  pose.header.stamp = img_time_stamp;
  
  pose.pose.position.x = tf(0,3);
  pose.pose.position.y = tf(1,3);
  pose.pose.position.z = tf(2,3);

  Eigen::Matrix3f rot = tf.block<3,3>(0,0);
  Eigen::Quaternionf q(rot);
  q.normalize();
  pose.pose.orientation.x = q.x();
  pose.pose.orientation.y = q.y();
  pose.pose.orientation.z = q.z();
  pose.pose.orientation.w = q.w();

  estimated_pose_pub.publish(pose);

  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(tf(0,3), tf(1,3), tf(2,3)));
  tf_transform.setBasis(tf::Matrix3x3(tf(0,0), tf(0,1), tf(0,2),
                                      tf(1,0), tf(1,1), tf(1,2),
                                      tf(2,0), tf(2,1), tf(2,2)));
  //br.sendTransform(tf::StampedTransform(tf_transform, img_time_stamp, "world", "camera"));
  br.sendTransform(tf::StampedTransform(tf_transform.inverse(), img_time_stamp, "camera", "object_pose"));
}

// decaying velocity model
void MapLocalizer::UpdateMotionModel(const Eigen::Matrix4f& oldTf, const Eigen::Matrix4f& newTf,
  const Eigen::Matrix<float, 6, 6>& cov, double dt)
{
  if(motion_model == "CONSTANT")
  {
    Eigen::Matrix4f new_from_old = newTf*oldTf.inverse();
    Eigen::Matrix4f cam_motion = new_from_old.log()/dt;
    Eigen::Matrix4f old_cam_vel = camera_velocity;
    camera_velocity = 0.9 * (0.5 * cam_motion + 0.5 * old_cam_vel);
  }
  else if(motion_model == "IMU")
  {
    // Apply correction measurement.  
    imu_mm->correct(newTf, cov);
  }
  else
  {
    camera_velocity = Eigen::MatrixXf::Zero(4,4);
  }
}

Eigen::Matrix4f MapLocalizer::ApplyMotionModel(double dt)
{
  if(motion_model == "IMU")
  {
    // Apply all IMU measurements since last ApplyMotionModel call
    return imu_mm->predict();
  }
  else if(motion_model == "CONSTANT")
  {
    return (dt*camera_velocity).exp()*currentPose;
  }
  else
  {
    return currentPose;
  }
}

void MapLocalizer::ResetMotionModel()
{
  if(motion_model == "CONSTANT")
  {
    camera_velocity = Eigen::MatrixXf::Zero(4,4);
  }
  else if(motion_model == "IMU")
  {
    imu_mm->reset();
  }
}

void MapLocalizer::spin(const ros::TimerEvent& e)
{
  PublishMap();

  ros::Time start;
  if(!get_frame && !get_virtual_image && !get_virtual_depth)
  {
    ros::Time current_time = ros::Time::now();
    double dt = (current_time - last_spin_time).toSec();
    last_spin_time = current_time;
    // Feature-based initialization with PnP localization (or edge tracking)
    if(localize_state == EDGES)
    {
      KeyframeContainer* kf = new KeyframeContainer(current_image, pnp_descriptor_type, false);
      ROS_INFO("Performing local Edge search...");
      start = ros::Time::now();
      Eigen::Matrix4f imgTf;
      if(!FindImageTfVirtualEdges(kf, ApplyMotionModel(dt), imgTf, true))
      //if(!FindImageTfVirtualEdges(kf, currentPose, imgTf, true))
      {
        ResetMotionModel();
        localize_state = PNP;
        delete kf;
        return;
      }
      ROS_INFO("FindImageTfVirtualEdges time: %f", (ros::Time::now()-start).toSec());  
      
      for(int i = 0; i < edge_tracking_iterations-1; i++)
      {
        Eigen::Matrix4f prevTf = imgTf;
        FindImageTfVirtualEdges(kf, prevTf, imgTf, true);
      }
      
      Eigen::Matrix<float, 6, 6> cov;
      UpdateMotionModel(currentPose, imgTf, cov, dt);

      currentPose = imgTf;
      UpdateVirtualSensorState(currentPose);
      PublishPose(currentPose);
      //std::cout << "Estimated tf: " << std::endl << imgTf << std::endl;
      //std::cout << "Actual tf: " << std::endl << kf->GetTf() << std::endl;
      ROS_INFO("Found image tf");
      delete kf;
    }
    else if(localize_state == PNP)
    {
      //start = ros::Time::now();
      KeyframeContainer* kf = new KeyframeContainer(current_image, pnp_descriptor_type, false);
      //ROS_INFO("Descriptor extraction time: %f", (ros::Time::now()-start).toSec());  
      
      ROS_INFO("Performing local PnP search...");
      Eigen::Matrix4f imgTf;
      

      ros::Time start = ros::Time::now();
      Eigen::Matrix<float, 6 ,6> cov;
      Eigen::Matrix4f currentPoseMM = ApplyMotionModel(dt);
      //std::cout << "currentPoseMM = " << std::endl << currentPoseMM << std::endl;
      //std::cout << "currentPose = " << std::endl << currentPose << std::endl;
      if(!FindImageTfVirtualPnp(kf, currentPoseMM, imgTf, pnp_descriptor_type, true, cov))
      {
        ResetMotionModel();
        numPnpRetrys++;
        if(numPnpRetrys > 1)
        {
          ROS_INFO("PnP failed, reinitializing using last known pose");
          numPnpRetrys = 0;
          localize_state = LOCAL_INIT;
        }
        delete kf;
        return;
      }
      UpdateMotionModel(currentPose, imgTf, cov, dt);
      numPnpRetrys = 0;
      ROS_INFO("FindImageTfVirtualPnp time: %f", (ros::Time::now()-start).toSec());  
      if(enable_edge_tracking && pnpReprojError < 1.5)
      {
        localize_state = EDGES;
      }
      currentPose = imgTf;
      UpdateVirtualSensorState(currentPose);
      PublishPose(currentPose);
      //std::cout << "Estimated tf: " << std::endl << imgTf << std::endl;
      //std::cout << "Actual tf: " << std::endl << kf->GetTf() << std::endl;
      ROS_INFO("Found image tf");
      delete kf;
    }
    else if (localize_state == INIT_PNP)
    {
      ROS_INFO("Refining matched pose with PnP...");
      Eigen::Matrix4f imgTf;
      
      start = ros::Time::now();
      KeyframeContainer* kf = new KeyframeContainer(current_image, img_match_descriptor_type);
      ROS_INFO("Descriptor extraction time: %f", (ros::Time::now()-start).toSec());  

      ros::Time start = ros::Time::now();
      Eigen::Matrix<float, 6 ,6> cov;
      if(!FindImageTfVirtualPnp(kf, currentPose, imgTf, img_match_descriptor_type, false, cov))
      {
        ROS_INFO("PnP init failed, reinitializing using last known pose");
        localize_state = LOCAL_INIT;
        
        delete kf;
        return;
      }

      ResetMotionModel();
      if(motion_model == "IMU")
      { 
        imu_mm->init(imgTf, cov);
      }

      numPnpRetrys = 0;
      localize_state = PNP;
      ROS_INFO("FindImageTfVirtualPnp time: %f", (ros::Time::now()-start).toSec());  
      currentPose = imgTf;
      UpdateVirtualSensorState(currentPose);
      PublishPose(currentPose);
      ROS_INFO("Found image tf");
      delete kf;
    }
    else
    {
      ros::Time start = ros::Time::now();
      Eigen::Matrix4f pose;
      bool localize_success;

      if(localize_state == LOCAL_INIT) 
      {
        localize_success = localization_init->localize(current_image, Kcv, &pose, &currentPose);
      }
      else if(localize_state == INIT) 
      {
        localize_success = localization_init->localize(current_image, Kcv, &pose);
      }

      if(localize_success)
      {
        ROS_INFO("Found image tf");
       
        localize_state = INIT_PNP;
        numLocalizeRetrys = 0;
        currentPose = pose;
        UpdateVirtualSensorState(currentPose);
        PublishPose(currentPose);
      }
      else
      {
        numLocalizeRetrys++;
        if(numLocalizeRetrys > 3)
        {
          ROS_INFO("Fully reinitializing");
          localize_state = INIT;
        }
      }

      ROS_INFO("LocalizationInit time: %f", (ros::Time::now()-start).toSec());  

    }
    ROS_INFO("Spin time: %f", (ros::Time::now() - spin_time).toSec());
    spin_time = ros::Time::now();

    get_frame = true;
  }
}


void MapLocalizer::PlotTf(Eigen::Matrix4f tf, std::string name)
{
  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(tf(0,3), tf(1,3), tf(2,3)));
  tf_transform.setBasis(tf::Matrix3x3(tf(0,0), tf(0,1), tf(0,2),
                                      tf(1,0), tf(1,1), tf(1,2),
                                      tf(2,0), tf(2,1), tf(2,2)));
  br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), "markers", name));
}

void MapLocalizer::PublishPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr pc)
{
  sensor_msgs::PointCloud2 pc2;
  pc2.header.frame_id = "/markers";
  pc2.header.stamp = ros::Time();

  pc->header = pcl_conversions::toPCL(pc2.header);
  pointcloud_pub.publish(pc);
}

void MapLocalizer::PublishPointCloud(const std::vector<pcl::PointXYZ>& pc)
{
  sensor_msgs::PointCloud2 pc2;
  pc2.header.frame_id = "/markers";
  pc2.header.stamp = ros::Time();

  pcl::PointCloud<pcl::PointXYZ> msg;
  msg.header = pcl_conversions::toPCL(pc2.header);

  for(unsigned int i = 0; i < pc.size(); i++)
  {
    msg.points.push_back(pc[i]);
  }

  pointcloud_pub.publish(msg);
}

void MapLocalizer::PublishMap()
{
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
  tf::Quaternion qtf;
  qtf.setRPY(0.0, 0, 0);
  transform.setRotation(qtf);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map_localize", "world"));
  
  tf::Transform marker_transform;
  marker_transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
  tf::Quaternion marker_qtf;
  marker_qtf.setRPY(0, 150.*(M_PI/180), 0);
  marker_transform.setRotation(marker_qtf);
  br.sendTransform(tf::StampedTransform(marker_transform, ros::Time::now(), "world", "markers"));

  visualization_msgs::Marker marker;
  marker.header.frame_id = "/world";
  marker.header.stamp = ros::Time();
  marker.ns = "map_localize";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0;
  marker.pose.orientation.y = 0;
  marker.pose.orientation.z = 0;
  marker.pose.orientation.w = 0;
  marker.scale.x = 1.;
  marker.scale.y = 1.;
  marker.scale.z = 1.;
  marker.color.a = 0.8;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  //only if using a MESH_RESOURCE marker type:
  marker.mesh_resource = std::string("package://map_localize") + mesh_filename;

  map_marker_pub.publish(marker);
}

Mat MapLocalizer::GetVirtualImageFromTopic(Mat& depths, Mat& mask)
{
  depths = Mat(virtual_height, virtual_width, CV_32F, Scalar(0));
  mask = Mat(virtual_height, virtual_width, CV_8U, Scalar(0));

  //std::cout << "step: " << depth_msg->step << " encoding: " << depth_msg->encoding << " bigendian: " << (depth_msg->is_bigendian ? 1 : 0) << std::endl;
  ros::Time start = ros::Time::now();
  #pragma omp parallel for
  for(int i = 0; i < virtual_height; i++)
  {
    for(int j = 0; j < virtual_width; j++)
    {
      union{
        float f;
        uchar b[4];
      } u;
 
      int index = i*current_virtual_depth_msg->step + j*(current_virtual_depth_msg->step/current_virtual_depth_msg->width);
      for(int k = 0; k < 4; k++)
      {
        u.b[k] = current_virtual_depth_msg->data[index+k];
      }
      if(u.f == u.f) // check if valid
      {
        depths.at<float>(i,j) = u.f;
        mask.at<uchar>(i,j) = 255;
      }
    }
  }
  ROS_INFO("VirtualImageFromTopic: parse depth time: %f", (ros::Time::now()-start).toSec());

  return current_virtual_image;
}


std::vector<int> MapLocalizer::FindPlaneInPointCloud(const std::vector<pcl::PointXYZ>& pts)
{
  std::vector<int> inliers;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);//, plane;

  cloud->points.resize(pts.size());
  for(unsigned int i = 0; i < pts.size(); i++)
  {
    cloud->points[i] = pts[i];
  }

  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
  ransac.setDistanceThreshold (.01);
  ransac.computeModel();
  ransac.getInliers(inliers);

  //pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *plane);
  return inliers;
}

std::vector<pcl::PointXYZ> MapLocalizer::GetPointCloudFromFrames(KeyframeContainer* kfc1, KeyframeContainer* kfc2)
{
  std::vector<CloudPoint> pcCv;	
  std::vector<pcl::PointXYZ> pc;	
  std::vector<KeyPoint> correspImg1Pt;
  const double matchRatio = 0.85;
	
  Eigen::Matrix4f tf1 = kfc1->GetTf().inverse();
  Eigen::Matrix4f tf2 = kfc2->GetTf().inverse();

  Matx34d P(tf1(0,0), tf1(0,1), tf1(0,2), tf1(0,3),
            tf1(1,0), tf1(1,1), tf1(1,2), tf1(1,3), 
            tf1(2,0), tf1(2,1), tf1(2,2), tf1(2,3));
  Matx34d P1(tf2(0,0), tf2(0,1), tf2(0,2), tf2(0,3),
             tf2(1,0), tf2(1,1), tf2(1,2), tf2(1,3), 
             tf2(2,0), tf2(2,1), tf2(2,2), tf2(2,3));
  Matx33d Kcv33(Kcv.at<double>(0,0), Kcv.at<double>(0,1), Kcv.at<double>(0,2),
                Kcv.at<double>(1,0), Kcv.at<double>(1,1), Kcv.at<double>(1,2),
                Kcv.at<double>(2,0), Kcv.at<double>(2,1), Kcv.at<double>(2,2));  

  //Find matches between kfc1 and kfc2
  FlannBasedMatcher matcher;
  std::vector < std::vector< DMatch > > matches;
  matcher.knnMatch( kfc1->GetDescriptors(), kfc2->GetDescriptors(), matches, 2 );

  std::vector< DMatch > goodMatches;
  std::vector< DMatch > allMatches;
  std::vector<Point3d> triangulatedPts1;
  std::vector<Point3d> triangulatedPts2;
  std::vector<KeyPoint> matchKps1;
  std::vector<KeyPoint> matchKps2;


  double reprojError;
  // Use ratio test to find good keypoint matches
  for(unsigned int j = 0; j < matches.size(); j++)
  {
    allMatches.push_back(matches[j][0]);
    if(matches[j][0].distance < matchRatio*matches[j][1].distance)
    {
      Point2f pt1 = kfc1->GetKeypoints()[matches[j][0].queryIdx].pt;
      Point2f pt2 = kfc2->GetKeypoints()[matches[j][0].trainIdx].pt;
      Mat_<double> triPt = LinearLSTriangulation(Point3d(pt1.x, pt1.y, 1), Kcv33*P, Point3d(pt2.x, pt2.y, 1), Kcv33*P1, &reprojError);
      //std::cout << "Reproj Error: " << *reprojError << std::endl;

      if(reprojError < 1.)
      {
        pc.push_back(pcl::PointXYZ(triPt(0), triPt(1), triPt(2)));
      
        goodMatches.push_back(matches[j][0]);
        matchKps1.push_back(kfc1->GetKeypoints()[matches[j][0].queryIdx]);
        matchKps2.push_back(kfc2->GetKeypoints()[matches[j][0].trainIdx]);
      }
    }
  }
  
 
#if 0
  namedWindow("matches", 1);
  Mat img_matches;
  drawMatches(kfc1->GetImage(), kfc1->GetKeypoints(), kfc2->GetImage(), kfc2->GetKeypoints(), goodMatches, img_matches);
  imshow("matches", img_matches);
  waitKey(0); 
#endif
  
  return pc;
}

std::vector<Point3d> MapLocalizer::PCLToPoint3d(const std::vector<pcl::PointXYZ>& cpvec)
{
  std::vector<Point3d> points;
  for(unsigned int i = 0; i < cpvec.size(); i++)
  {
    Point3d pt(cpvec[i].x, cpvec[i].y, cpvec[i].z);
    points.push_back(pt);  
  }
  return points;
}

void MapLocalizer::ReprojectMask(Mat& dst, const Mat& src, const Eigen::Matrix3f& dstK, 
  const Eigen::Matrix3f& srcK)
{
  double fxs = srcK(0,0);
  double fys = srcK(1,1);
  double cxs = srcK(0,2);
  double cys = srcK(1,2);
  double fxd = dstK(0,0);
  double fyd = dstK(1,1);
  double cxd = dstK(0,2);
  double cyd = dstK(1,2);
  for(int i = 0; i < src.rows; i++)
  {
    for(int j = 0; j < src.cols; j++)
    {
      if(src.at<uchar>(i,j) != 255)
        continue;
      double px_un = (j - cxs)/fxs;
      double py_un = (i - cys)/fys;

      int dst_x = floor(px_un*fxd + cxd);
      int dst_y = floor(py_un*fyd + cyd);
      if(dst_x < 0 || dst_x >= dst.cols || dst_y < 0 || dst_y >= dst.rows)
        continue;
      
      dst.at<uchar>(dst_y, dst_x) = src.at<uchar>(i,j);
    }
  }
  
  medianBlur(dst, dst, 3);
}  



bool MapLocalizer::FindImageTfVirtualEdges(KeyframeContainer* kfc, Eigen::Matrix4f vimgTf, Eigen::Matrix4f& tf, bool mask_kf)
{
  tf = Eigen::MatrixXf::Identity(4,4);

  // Get virtual image and depth map
  Mat depth, mask;
  Mat vimg, vimg_masked;
  Eigen::Matrix3f vimgK;
  ros::Time start = ros::Time::now();
  if(virtual_image_source == "gazebo")
  {
    vimgK = virtual_K; 
    vimg = GetVirtualImageFromTopic(depth, mask);
  }
  else if(virtual_image_source == "ogre" || virtual_image_source == "point_cloud")
  {
    vimgK = vig->GetK(); 
    vimg = vig->GenerateVirtualImage(vimgTf, depth, mask);
  }
  else
  {
    ROS_ERROR("Invalid virtual_image_source");
    return false;
  }
  vimg.copyTo(vimg_masked, mask);
  ROS_INFO("VirtualEdges: generate virtual img time: %f", (ros::Time::now()-start).toSec());
  
  Mat kf_mask;
  if(mask_kf)
  {
    start = ros::Time::now();
    Mat reproj_mask = Mat(kfc->GetImage().rows, kfc->GetImage().cols, CV_8U, Scalar(0));
    ReprojectMask(reproj_mask, mask, K_scaled, vimgK);

    //dilate mask so as not to mask good features that may have moved
    int dilate_size = 15;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*dilate_size+1,2*dilate_size+1), Point(dilate_size,dilate_size));
    dilate(reproj_mask, kf_mask, element);
    kfc->SetMask(kf_mask);
    
    ROS_INFO("VirtualEdges: Reproject mask time: %f", (ros::Time::now()-start).toSec());  
    //namedWindow( "Reproj Mask", WINDOW_NORMAL );// Create a window for display.
    //imshow( "Reproj Mask", reproj_mask ); 
    if(show_debug)
    {
      Mat query_masked;
      kfc->GetImage().copyTo(query_masked, kf_mask);
      namedWindow( "Query Masked", WINDOW_NORMAL );// Create a window for display.
      imshow( "Query Masked", query_masked ); 
    }
  }
  

  start = ros::Time::now();
  std::vector<EdgeTrackingUtil::SamplePoint> sps = 
    EdgeTrackingUtil::getEdgeMatches(vimg_masked, kfc->GetImage(), vimgK, K_scaled, depth, 
      kf_mask, vimgTf);
  ROS_INFO("VirtualEdges: Edge matching time: %f", (ros::Time::now()-start).toSec());  

  double avgError = 0;
  for(int i = 0; i < sps.size(); i++)
  {
    avgError += sps[i].dist;
  }
  avgError /= sps.size();

  // hacky way to detect failure 
  if(avgError > 15 || sps.size() < 15)
    return false;
  
  ROS_INFO("VirtualEdges: avg matching error: %f", avgError);
  start = ros::Time::now();
  //EdgeTrackingUtil::getEstimatedPosePnP(tf, vimgTf.inverse(), sps, Kcv);
  EdgeTrackingUtil::getEstimatedPoseIRLS(tf, vimgTf.inverse(), sps, K_scaled);
  tf = tf.inverse();
  ROS_INFO("VirtualEdges: IRLS time: %f", (ros::Time::now()-start).toSec());  

  return true;
}



bool MapLocalizer::FindImageTfVirtualPnp(KeyframeContainer* kfc, Eigen::Matrix4f vimgTf, Eigen::Matrix4f& tf, std::string vdesc_type, bool mask_kf, Eigen::Matrix<float, 6, 6>& cov)
{
  tf = Eigen::MatrixXf::Identity(4,4);

  // Get virtual image and depth map
  Mat depth, mask;
  Mat vimg;
  Eigen::Matrix3f vimgK, vimgK_inv;
  ros::Time start = ros::Time::now();
  if(virtual_image_source == "gazebo")
  {
    vimgK = virtual_K; 
    vimg = GetVirtualImageFromTopic(depth, mask);
  }
  else if(virtual_image_source == "ogre" || virtual_image_source == "point_cloud")
  {
    vimgK = vig->GetK(); 
    vimg = vig->GenerateVirtualImage(vimgTf, depth, mask);
  }
  else
  {
    ROS_ERROR("Invalid virtual_image_source");
    return false;
  }
  vimgK_inv = vimgK.inverse();
  ROS_INFO("VirtualPnP: generate virtual img time: %f", (ros::Time::now()-start).toSec());
  
  if(mask_kf)
  {
    Mat reproj_mask = Mat(kfc->GetImage().rows, kfc->GetImage().cols, CV_8U, Scalar(0));
    start = ros::Time::now();
    ReprojectMask(reproj_mask, mask, K_scaled, vimgK);
    int dilate_size = 15;
    Mat element = getStructuringElement(MORPH_RECT, Size(2*dilate_size+1,2*dilate_size+1), Point(dilate_size,dilate_size));
    dilate(reproj_mask, reproj_mask, element);
    ROS_INFO("VirtualPnP: reproject mask time: %f", (ros::Time::now()-start).toSec());
    kfc->SetMask(reproj_mask);
    
    start = ros::Time::now();
    kfc->ExtractFeatures();
    ROS_INFO("Descriptor extraction time: %f", (ros::Time::now()-start).toSec());  
    //namedWindow( "Reproj Mask", WINDOW_NORMAL );// Create a window for display.
    //imshow( "Reproj Mask", reproj_mask ); 
    if(show_debug)
    {
      Mat query_masked;
      kfc->GetImage().copyTo(query_masked, reproj_mask);
      namedWindow( "Query Masked", WINDOW_NORMAL );// Create a window for display.
      imshow( "Query Masked", query_masked ); 
    }
  }
  if(show_debug)
  {
    Mat depth_im;
    double min_depth, max_depth;
    minMaxLoc(depth, &min_depth, &max_depth);    
    //std::cout << "min_depth=" << min_depth << " max_depth=" << max_depth << std::endl;
    depth.convertTo(depth_im, CV_8U, 255.0/(max_depth-min_depth), 0);// -min_depth*255.0/(max_depth-min_depth));

    namedWindow( "Query", WINDOW_NORMAL );// Create a window for display.
    imshow( "Query", kfc->GetImage() ); 
    namedWindow( "Virtual", WINDOW_NORMAL );// Create a window for display.
    imshow( "Virtual", vimg ); 
    namedWindow( "Depth", WINDOW_NORMAL );// Create a window for display.
    imshow( "Depth", depth_im ); 
    //namedWindow( "Mask", WINDOW_NORMAL );// Create a window for display.
    //imshow( "Mask", mask ); 
    waitKey(1);
  }

  // Find features in virtual image
  std::vector<KeyPoint> vkps;
  Mat vdesc;
  gpu::GpuMat vdesc_gpu;
  std::vector < std::vector< DMatch > > matches;
  
  // Find image features matches between kfc and vimg
  double matchRatio;

  start = ros::Time::now();
  if(vdesc_type == "asift")
  {
    ASiftDetector detector;
    detector.detectAndCompute(vimg, vkps, vdesc, mask, ASiftDetector::SIFT);

    matchRatio = 0.7;
  }
  else if(vdesc_type == "asurf")
  {
    ASiftDetector detector;
    detector.detectAndCompute(vimg, vkps, vdesc, mask, ASiftDetector::SURF);

    matchRatio = 0.7;
  }
  else if(vdesc_type == "orb")
  {
    ORB orb(1000, 1.2f, 4);
    orb(vimg, mask, vkps, vdesc);

    std::cout << "vkps: " << vkps.size() << std::endl;

    matchRatio = 0.8;
  }
  else if(vdesc_type == "surf")
  {
    SurfFeatureDetector detector;
    detector.detect(vimg, vkps, mask);

    SurfDescriptorExtractor extractor;
    extractor.compute(vimg, vkps, vdesc);

    matchRatio = 0.7;
  }
  else if(vdesc_type == "surf_gpu")
  {
    gpu::SURF_GPU surf_gpu;
   
    if(virtual_image_source == "gazebo")
      cvtColor(vimg, vimg, CV_BGR2GRAY);
    gpu::GpuMat vkps_gpu, mask_gpu(mask), vimg_gpu(vimg);
    
    surf_gpu(vimg_gpu, mask_gpu, vkps_gpu, vdesc_gpu);
    surf_gpu.downloadKeypoints(vkps_gpu, vkps);   

    std::cout << "vkps: " << vkps.size() << std::endl;
 
    matchRatio = 0.8;
  }

  if(vkps.size() <= 0)
  {
    ROS_WARN("No keypoints found in virtual image");
    return false;
  }

  if(pnp_match_radius > 0 && vdesc_type == "orb")
  {
    std::vector<KeyPoint> kf_kps = kfc->GetKeypoints();
    Mat kf_desc = kfc->GetDescriptors();
    int step = kf_desc.step / sizeof(kf_desc.ptr()[0]);
    for(int i = 0; i < vkps.size(); i++)
    {
      int best_match_idx = -1;
      int best_match_dist = -1;
      Eigen::Vector3f vkp_in_kf(vkps[i].pt.x, vkps[i].pt.y, 1);
      vkp_in_kf = K_scaled*vimgK_inv*vkp_in_kf;
      vkp_in_kf /= vkp_in_kf(2);
      for(int j = 0; j < kf_kps.size(); j++)
      {
        if(sqrt(pow(vkp_in_kf(0)-kf_kps[j].pt.x,2) + pow(vkp_in_kf(1)-kf_kps[j].pt.y,2)) > pnp_match_radius)
        {
          continue;
        }
        int dist = cv::normHamming(vdesc.ptr(i), kf_desc.ptr() + step*j, kf_desc.cols);
        if(dist < best_match_dist || best_match_dist == -1)
        {
          best_match_dist = dist;
          best_match_idx = j;
        }
      }
      if(best_match_idx != -1)
      {
        std::vector<DMatch> pmatches(2);
        pmatches[0] = DMatch(best_match_idx, i, best_match_dist);
        pmatches[1] = DMatch(best_match_idx, i, std::numeric_limits<float>::max());
        matches.push_back(pmatches);
      }
    }
  }
  else
  {
    // TODO: Add option to match all descriptors on GPU
    if(vdesc_type == "surf_gpu")
    {
      gpu::BFMatcher_GPU matcher;
      matcher.knnMatch(kfc->GetGPUDescriptors(), vdesc_gpu, matches, 2);  
    }
    else if(vdesc_type == "orb")
    {
      BFMatcher matcher(NORM_HAMMING);
      matcher.knnMatch( kfc->GetDescriptors(), vdesc, matches, 2 );
    }
    else
    {
      FlannBasedMatcher matcher;
      matcher.knnMatch( kfc->GetDescriptors(), vdesc, matches, 2 );
    }
  }

  ROS_INFO("VirtualPnP: find keypoints/matches time: %f", (ros::Time::now()-start).toSec());

  std::vector< DMatch > goodMatches;
  std::vector< DMatch > allMatches;
  std::vector<Point2f> matchPts;
  std::vector<Point2f> matchPts3dProj;
  std::vector<Point3f> matchPts3d;
  std::vector<pcl::PointXYZ> matchPts3d_pcl;
  
  start = ros::Time::now();
  for(unsigned int j = 0; j < matches.size(); j++)
  {
    allMatches.push_back(matches[j][0]);
    if(matches[j][0].distance < matchRatio*matches[j][1].distance)
    {
      // Back-project point to 3d
      if(matches[j][0].trainIdx >= vkps.size() || matches[j][0].queryIdx >= kfc->GetKeypoints().size())
      {
        std::cout <<  "Index mismatch? AHH: " << matches[j][0].trainIdx << " " << matches[j][0].queryIdx << " " << vkps.size() << " " << kfc->GetKeypoints().size() << std::endl;
      }

      Point2f kp = vkps[matches[j][0].trainIdx].pt;
      Eigen::Vector3f hkp(kp.x, kp.y, 1);
      Eigen::Vector3f backproj = vimgK.inverse()*hkp;
      backproj /= backproj(2);    
      backproj *= depth.at<float>(kp.y, kp.x);
      Eigen::Vector4f backproj_h(backproj(0), backproj(1), backproj(2), 1);
      backproj_h = vimgTf*backproj_h;
 
      goodMatches.push_back(matches[j][0]);
      matchPts3dProj.push_back(kp);
      matchPts.push_back(kfc->GetKeypoints()[matches[j][0].queryIdx].pt);
      matchPts3d.push_back(Point3f(backproj_h(0), backproj_h(1), backproj_h(2)));
      matchPts3d_pcl.push_back(pcl::PointXYZ(backproj_h(0), backproj_h(1), backproj_h(2)));
    }
  } 
  ROS_INFO("VirtualPnP: match filter time: %f", (ros::Time::now()-start).toSec());

  if(show_pnp_matches)
  { 
    //PublishPointCloud(matchPts3d_pcl);
    Mat img_matches;
    drawMatches(kfc->GetImage(), kfc->GetKeypoints(), vimg, vkps, goodMatches, img_matches);
    imshow("PnP Matches", img_matches);
    waitKey(1);
  }

  if(goodMatches.size() < 4)
  {
    ROS_WARN("Not enough matches found in virtual image");
    return false;
  }

  /**** Pnp on known correspondences from virtual image ****  
  Mat Rvec_true, t_true;
  solvePnP(matchPts3d, matchPts3dProj, Kcv, distcoeffcv, Rvec_true, t_true);

  Mat Rtrue;
  Rodrigues(Rvec_true, Rtrue);
  Eigen::Matrix4f true_tf;
  true_tf << Rtrue.at<double>(0,0), Rtrue.at<double>(0,1), Rtrue.at<double>(0,2), t_true.at<double>(0),
        Rtrue.at<double>(1,0), Rtrue.at<double>(1,1), Rtrue.at<double>(1,2), t_true.at<double>(1),
        Rtrue.at<double>(2,0), Rtrue.at<double>(2,1), Rtrue.at<double>(2,2), t_true.at<double>(2),
             0,      0,      0,    1;
  std::cout << "Known: " << std::endl << vimgTf << std::endl << std::endl << true_tf.inverse() << std::endl; 
  *****/


  Eigen::Matrix4f tfran;
  //solvePnPRansac(matchPts3d, matchPts, Kcv, 
  std::vector<int> inlierIdx;
  start = ros::Time::now();
  if(!PnPUtil::RansacPnP(matchPts3d, matchPts, Kcv, vimgTf.inverse(), tfran, inlierIdx, &pnpReprojError, &cov))
  {
    return false;
  }
 
  // compute covariance of inverse transform from transform;
  Eigen::Matrix<float, 6, 6> J;
  J.setZero();
  J.block<3,3>(0,0) = -Eigen::MatrixXf::Identity(3,3);
  J.block<3,3>(3,3) = -tfran.block<3,3>(0,0).transpose();
  Eigen::Matrix3d A;
  gcop::SO3::Instance().hat(A, (-tfran.block<3,3>(0,0).transpose()*tfran.block<3,1>(0,3)).cast<double>()); // hat(-R^Tt)
  J.block<3,3>(3,0) = A.cast<float>();

  cov = J*pixel_noise*cov*J.transpose();
  //std::cout << "R, t inv covariance:" << std::endl << cov << std::endl;

  if(show_pnp_matches)
  { 
    std::vector< DMatch > inlierMatches;
    for(int j = 0; j < inlierIdx.size(); j++)
    {
      inlierMatches.push_back(goodMatches[inlierIdx[j]]);
    }
    Mat img_matches;
    drawMatches(kfc->GetImage(), kfc->GetKeypoints(), vimg, vkps, inlierMatches, img_matches);
    imshow("PnP Match Inliers", img_matches);
    waitKey(1);
  }

  ROS_INFO("VirtualPnP: Ransac PnP time: %f", (ros::Time::now()-start).toSec());
  ROS_INFO("VirtualPnP: found match. Average reproj error = %f", pnpReprojError);
  tf = tfran.inverse();
  return true;
}


Eigen::Matrix4f MapLocalizer::FindImageTfPnp(KeyframeContainer* kfc, const MapFeatures& mf)
{
  Eigen::Matrix4f tf;
  
  // Find image features matches in map
  const double matchRatio = 0.6;

  FlannBasedMatcher matcher;
  std::vector < std::vector< DMatch > > matches;
  matcher.knnMatch( kfc->GetDescriptors(), mf.GetDescriptors(), matches, 2 );

  std::vector< DMatch > goodMatches;
  std::vector< DMatch > allMatches;
  std::vector<Point2f> matchPts;
  std::vector<Point3f> matchPts3d;

  for(unsigned int j = 0; j < matches.size(); j++)
  {
    allMatches.push_back(matches[j][0]);
    if(matches[j][0].distance < matchRatio*matches[j][1].distance)
    {
      pcl::PointXYZ pt3d = mf.GetKeypoints()[matches[j][0].trainIdx];

      goodMatches.push_back(matches[j][0]);
      matchPts.push_back(kfc->GetKeypoints()[matches[j][0].queryIdx].pt);
      matchPts3d.push_back(Point3f(pt3d.x, pt3d.y, pt3d.z));
    }
  }
 
  if(goodMatches.size() <= 0)
  {
    ROS_WARN("No matches found in map");
    return tf;
  }
  // Solve for camera transform
  Mat Rvec, t;
  //solvePnP(matchPts3d, matchPts, Kcv, distcoeffcv, Rvec, t);
  solvePnPRansac(matchPts3d, matchPts, Kcv, distcoeffcv, Rvec, t);

  Mat R;
  Rodrigues(Rvec, R);

  tf << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2),
             0,      0,      0,    1;
  return tf.inverse();
}


