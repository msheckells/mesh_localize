#include "map_localize/MapLocalizer.h"
#include <tinyxml.h>
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

#include "map_localize/FindCameraMatrices.h"
#include "map_localize/Triangulation.h"
#include "map_localize/ASiftDetector.h"
#include "map_localize/lsh.hpp"
#include "map_localize/FeatureMatchLocalizer.h"
#include "map_localize/FABMAPLocalizer.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "gazebo_msgs/LinkState.h"
#include "gazebo_msgs/SetLinkState.h"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>

//#include <kvld/kvld.h>

#define SHOW_MATCHES_ 

MapLocalizer::MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private):
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
  if (!nh_private.getParam ("show_global_matches", show_global_matches))
    show_global_matches = false;
  if (!nh_private.getParam("descriptor_filename", descriptor_filename))
    descriptor_filename = "";
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
  

  // TODO: read K and distcoeff from param file
  K << 701.907522339299, 0, 352.73599016194, 0, 704.43277859417, 230.636873050629, 0, 0, 1;
  distcoeff = Eigen::VectorXf(5);
  distcoeff << -0.456758192707853, 0.197636354824418, 0.000543685887014507, 0.000401738655456894, 0;

  Kcv_undistort = Matx33d(K(0,0), K(0,1), K(0,2),
              K(1,0), K(1,1), K(1,2),
              K(2,0), K(2,1), K(2,2)); 
  Kcv = image_scale*Kcv_undistort;
  Kcv(2,2) = 1;

  distcoeffcv = (Mat_<double>(5,1) << distcoeff(0), distcoeff(1), distcoeff(2), distcoeff(3), distcoeff(4)); 

  map_K << 1799.352269, 0, 1799.029749, 0, 1261.4382272, 957.3402899, 0, 0, 1;
  map_distcoeff = Eigen::VectorXf(5);
  map_distcoeff << 0, 0, 0, 0, 0;
  //map_distcoeff << -.0066106, .04618129, -.00042169, -.004390247, -.048470351;

  map_Kcv = Matx33d(map_K(0,0), map_K(0,1), map_K(0,2),
              map_K(1,0), map_K(1,1), map_K(1,2),
              map_K(2,0), map_K(2,1), map_K(2,2)); 
  map_distcoeffcv = (Mat_<double>(5,1) << map_distcoeff(0), map_distcoeff(1), map_distcoeff(2), map_distcoeff(3), map_distcoeff(4)); 

  if(image_scale != 1.0)
  {
    ROS_INFO("Scaling images by %f", image_scale);
  }

  ROS_INFO("Using %s for pnp descriptors and %s for image matching descriptors", pnp_descriptor_type.c_str(), img_match_descriptor_type.c_str());


  if(!LoadPhotoscanFile(photoscan_filename))
  {
    return;
  }

  if(global_localization_alg == "feature_match")
  {
    ROS_INFO("Using feature matching for initialization");
    localization_init = new FeatureMatchLocalizer(cameras, img_match_descriptor_type, show_global_matches, load_descriptors, descriptor_filename);
  }
  else if(global_localization_alg == "fabmap")
  {
    ROS_INFO("Using OpenFABMAP for initialization");
    if(img_match_descriptor_type != "surf")
    {
      ROS_ERROR("img_match_descriptor_type must be 'surf' when using OpenFABMAP");
      return;
    }
    localization_init = new FABMAPLocalizer(cameras, img_match_descriptor_type, show_global_matches, load_descriptors, descriptor_filename);
  }
  else
  {
    ROS_ERROR("%s is not a valid initialization option", global_localization_alg.c_str());
    return; 
  }
  //std::cout << "Mapping features to point cloud..." << std::flush;
  //map_features = MapFeatures(keyframes, map_cloud);
  //std::cout << "done" << std::endl;

  std::srand(time(NULL));
 
  estimated_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/map_localize/estimated_pose", 1);
  map_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/map", 1);
  match_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/match_points", 1);
  tvec_marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/map_localize/t_vectors", 1);
  epos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/estimated_position", 1);
  apos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/actual_position", 1);
  path_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/estimated_path", 1);
  pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/map_localize/pointcloud", 1);

  image_sub = nh.subscribe<sensor_msgs::Image>("image", 1, &MapLocalizer::HandleImage, this, ros::TransportHints().tcpNoDelay());


  if(virtual_image_source == "point_cloud")
  {
    ROS_INFO("Using PCL point cloud for virtual image generation");
    map_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    ROS_INFO("Loading point cloud %s", pc_filename.c_str());
    if(pcl::io::loadPCDFile<pcl::PointXYZRGBNormal> (pc_filename, *map_cloud) == -1)
    {
      std::cout << "Could not open point cloud " << pc_filename << std::endl;
      return;
    }
    ROS_INFO("Successfully loaded point cloud");
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
    virtual_Kcv = Matx33d(virtual_K(0,0), virtual_K(0,1), virtual_K(0,2),
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

  localize_state = INIT;

  spin_time = ros::Time::now();
  timer = nh_private.createTimer(ros::Duration(0.04), &MapLocalizer::spin, this);
}

MapLocalizer::~MapLocalizer()
{
  for(unsigned int i = 0; i < keyframes.size(); i++)
  {
    delete keyframes[i];
  }
  keyframes.clear();
  for(unsigned int i = 0; i < cameras.size(); i++)
  {
    delete cameras[i];
  }
  cameras.clear();
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
    undistort(cvImg->image, current_image, Kcv_undistort, distcoeffcv);
    if(image_scale != 1.0)
    {
      resize(current_image, current_image, Size(0,0), image_scale, image_scale);
    }
    ROS_INFO("Image copy time: %f", (ros::Time::now()-start).toSec());  
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
  br.sendTransform(tf::StampedTransform(tf_transform, img_time_stamp, "world", "estimated_pelican"));
}

void MapLocalizer::spin(const ros::TimerEvent& e)
{
  PublishMap();

  ros::Time start;
  if(!get_frame && !get_virtual_image && !get_virtual_depth)
  {
    //Mat test = imread("/home/matt/uav_image_data/run11/frame0029.jpg", CV_LOAD_IMAGE_GRAYSCALE );
    //KeyframeContainer* kf = new KeyframeContainer(test, Eigen::Matrix4f());
    //KeyframeContainer* kf = keyframes[std::rand() % keyframes.size()];

#if 1
    // ASift initialization with PnP localization
    if(localize_state == PNP)
    {
      start = ros::Time::now();
      KeyframeContainer* kf = new KeyframeContainer(current_image, pnp_descriptor_type);
      ROS_INFO("Descriptor extraction time: %f", (ros::Time::now()-start).toSec());  
      
      ROS_INFO("Performing local PnP search...");
      Eigen::Matrix4f imgTf;
      
      ros::Time start = ros::Time::now();
      if(!FindImageTfVirtualPnp(kf, currentPose, virtual_K, imgTf, pnp_descriptor_type))
      {
        numPnpRetrys++;
        if(numPnpRetrys > 1)
        {
          numPnpRetrys = 0;
          localize_state = LOCAL_INIT;
        }
        delete kf;
        return;
      }
      numPnpRetrys = 0;
      ROS_INFO("FindImageTfVirtualPnp time: %f", (ros::Time::now()-start).toSec());  
      currentPose = imgTf;
      UpdateVirtualSensorState(currentPose);
      PublishPose(currentPose);
      //std::cout << "Estimated tf: " << std::endl << imgTf << std::endl;
      //std::cout << "Actual tf: " << std::endl << kf->GetTf() << std::endl;
      positionList.push_back(imgTf.block<3,1>(0,3));
      PublishTfViz(imgTf, kf->GetTf());
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
      if(!FindImageTfVirtualPnp(kf, currentPose, virtual_K, imgTf, img_match_descriptor_type))
      {
        localize_state = LOCAL_INIT;
        
        delete kf;
        return;
      }

      numPnpRetrys = 0;
      localize_state = PNP;
      ROS_INFO("FindImageTfVirtualPnp time: %f", (ros::Time::now()-start).toSec());  
      currentPose = imgTf;
      UpdateVirtualSensorState(currentPose);
      PublishPose(currentPose);
      positionList.push_back(imgTf.block<3,1>(0,3));
      PublishTfViz(imgTf, kf->GetTf());
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
        localize_success = localization_init->localize(current_image, &pose, &currentPose);
      }
      else if(localize_state == INIT) 
      {
        localize_success = localization_init->localize(current_image, &pose);
      }

      if(localize_success)
      {
        ROS_INFO("Found image tf");
       
        localize_state = INIT_PNP;
        numLocalizeRetrys = 0;
        currentPose = pose;
        UpdateVirtualSensorState(currentPose);
        PublishPose(currentPose);
        positionList.push_back(currentPose.block<3,1>(0,3));
        //PublishTfViz(currentPose, kf->GetTf());
      }
      else
      {
        numLocalizeRetrys++;
        if(numLocalizeRetrys > 3)
        {
          localize_state = INIT;
        }
      }

      ROS_INFO("LocalizationInit time: %f", (ros::Time::now()-start).toSec());  

    }
#else
    // ASift localization with local search refinement.  Uses fundamental matrix instead of Pnp
    std::vector< KeyframeMatch > matches = FindImageMatches(kf, 7);//, isLocalized); // Only do local search if position is known
    std::vector< KeyframeMatch > goodMatches;
    std::vector< Eigen::Vector3f > goodTVecs;

    for(unsigned int i = 0; i < matches.size(); i++)
    {
      if(matches[i].matchKps1.size() == 0 || matches[i].matchKps2.size() == 0)
      {
        ROS_INFO("Match had no keypoints");
        
        delete currentKeyframe;
        currentKeyframe = NULL;
        return;
      }
    }
    //ROS_INFO("Found matches");
    //std::vector<pcl::PointXYZ> pclCloud = GetPointCloudFromFrames(matches[1].kfc, matches[2].kfc);
    //std::vector<Point3d> cvCloud = PCLToPoint3d(pclCloud);
    //std::vector<int> planeIdx, nonplaneIdx;
    //ROS_INFO("Got point cloud");
    /*
    if(pclCloud.size() > 5)
    {
      std::vector<int> inliers = FindPlaneInPointCloud(pclCloud);
      std::vector<pcl::PointXYZ> planeCloud;
      for(unsigned int i = 0; i < inliers.size(); i++)
      {
        planeCloud.push_back(pclCloud[inliers[i]]);
      }
      //PublishPointCloud(map_cloud);
      //PublishPointCloud(pclCloud);
      TestCoplanarity(cvCloud, planeIdx, nonplaneIdx);
      //ROS_INFO("Tested for planes");
    }
    else
    {
      //ROS_INFO("No point cloud available");
    }
    PlotTf(matches[1].kfc->GetTf(), "match1");
    PlotTf(matches[2].kfc->GetTf(), "match2");
    */
#ifdef SHOW_MATCHES_
    namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Query", kf->GetImage() ); 
    waitKey(0);
    for(int i = 0; i < matches.size(); i++)
    {
      Mat img_matches;
      //drawMatches(kf->GetImage(), kf->GetKeypoints(), matches[i].kfc->GetImage(), matches[i].kfc->GetKeypoints(), matches[i].matches, img_matches);
      //imshow("matches", img_matches);
      namedWindow( "Match", WINDOW_AUTOSIZE );// Create a window for display.
      imshow( "Match", matches[i].kfc->GetImage() ); 
      waitKey(0);
    }
#endif

    Eigen::Matrix4f imgTf = FindImageTfSfm(kf, matches, goodMatches, goodTVecs);
    if(goodMatches.size() >= 2)
    {
      ROS_INFO("Found image tf");
      isLocalized = true;
      numLocalizeRetrys = 0;
      currentPosition = imgTf.block<3,1>(0,3);
      positionList.push_back(currentPosition);
      PublishTfViz(imgTf, kf->GetTf());
      PublishSfmMatchViz(goodMatches, goodTVecs);
    }
    else
    {
      numLocalizeRetrys++;
      if(numLocalizeRetrys > 5)
      {
        isLocalized = false;
      }
    }
#endif
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

void MapLocalizer::PublishSfmMatchViz(std::vector<KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs)
{
  const double ptSize = 0.1;

  std::vector<geometry_msgs::Point> match_pos;
  for(unsigned int i = 0; i < matches.size(); i++)
  {
    geometry_msgs::Point pt;
    pt.x = matches[i].kfc->GetTf()(0,3);
    pt.y = matches[i].kfc->GetTf()(1,3);
    pt.z = matches[i].kfc->GetTf()(2,3);
    match_pos.push_back(pt);
  }
  visualization_msgs::Marker match_marker;

  match_marker.header.frame_id = "/markers";
  match_marker.header.stamp = ros::Time();
  match_marker.ns = "map_localize";
  match_marker.id = 1;
  match_marker.type = visualization_msgs::Marker::POINTS;
  match_marker.action = visualization_msgs::Marker::ADD;
  match_marker.pose.position.x = 0;
  match_marker.pose.position.y = 0;
  match_marker.pose.position.z = 0;
  match_marker.pose.orientation.x = 0;
  match_marker.pose.orientation.y = 0;
  match_marker.pose.orientation.z = 0;
  match_marker.pose.orientation.w = 0;
  match_marker.scale.x = ptSize;
  match_marker.scale.y = ptSize;
  match_marker.scale.z = ptSize;
  match_marker.color.a = 1.0;
  match_marker.color.r = 1.0;
  match_marker.color.g = 0.0;
  match_marker.color.b = 0.0;
  match_marker.points = match_pos;

  match_marker_pub.publish(match_marker);

  
  std::vector<visualization_msgs::Marker> tvec_markers_toremove;
  for(unsigned int i = 0; i < 30; i++)
  {
    visualization_msgs::Marker tvec_marker;
    tvec_marker.header.frame_id = "/markers";
    tvec_marker.header.stamp = ros::Time();
    tvec_marker.ns = "map_localize";
    tvec_marker.id = 2+i;
    tvec_marker.type = visualization_msgs::Marker::ARROW;
    tvec_marker.action = visualization_msgs::Marker::DELETE;
    
    tvec_markers_toremove.push_back(tvec_marker);
  }
  visualization_msgs::MarkerArray tvec_array_toremove;
  tvec_array_toremove.markers = tvec_markers_toremove;

  tvec_marker_pub.publish(tvec_array_toremove);

  std::vector<visualization_msgs::Marker> tvec_markers;
  for(unsigned int i = 0; i < matches.size(); i++)
  {
    visualization_msgs::Marker tvec_marker;
    const double len = 5.0;
    std::vector<geometry_msgs::Point> vec_pos;
    geometry_msgs::Point endpt;
    endpt.x = match_pos[i].x + len*tvecs[i](0);
    endpt.y = match_pos[i].y + len*tvecs[i](1);
    endpt.z = match_pos[i].z + len*tvecs[i](2);
    vec_pos.push_back(match_pos[i]);
    vec_pos.push_back(endpt);

    tvec_marker.header.frame_id = "/markers";
    tvec_marker.header.stamp = ros::Time();
    tvec_marker.ns = "map_localize";
    tvec_marker.id = 2+i;
    tvec_marker.type = visualization_msgs::Marker::ARROW;
    tvec_marker.action = visualization_msgs::Marker::ADD;
    tvec_marker.pose.position.x = 0; 
    tvec_marker.pose.position.y = 0; 
    tvec_marker.pose.position.z = 0; 
    tvec_marker.pose.orientation.x = 0;
    tvec_marker.pose.orientation.y = 0;
    tvec_marker.pose.orientation.z = 0;
    tvec_marker.pose.orientation.w = 0;
    tvec_marker.scale.x = 0.08;
    tvec_marker.scale.y = 0.11;
    tvec_marker.color.a = 1.0;
    tvec_marker.color.r = 1.0;
    tvec_marker.color.g = 0.0;
    tvec_marker.color.b = 0.0;
    tvec_marker.points = vec_pos;
  
    tvec_markers.push_back(tvec_marker);
  }

  visualization_msgs::MarkerArray tvec_array;
  tvec_array.markers = tvec_markers;

  tvec_marker_pub.publish(tvec_array);

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

void MapLocalizer::PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf)
{
  const double ptSize = 0.1;

  std::vector<geometry_msgs::Point> epos;
  geometry_msgs::Point epos_pt;
  epos_pt.x = imgTf(0,3);
  epos_pt.y = imgTf(1,3);
  epos_pt.z = imgTf(2,3);
  epos.push_back(epos_pt);
  visualization_msgs::Marker epos_marker;
 
  epos_marker.header.frame_id = "/markers";
  epos_marker.header.stamp = ros::Time();
  epos_marker.ns = "map_localize";
  epos_marker.id = 900;
  epos_marker.type = visualization_msgs::Marker::POINTS;
  epos_marker.action = visualization_msgs::Marker::ADD;
  epos_marker.pose.position.x = 0;
  epos_marker.pose.position.y = 0;
  epos_marker.pose.position.z = 0;
  epos_marker.pose.orientation.x = 0;
  epos_marker.pose.orientation.y = 0;
  epos_marker.pose.orientation.z = 0;
  epos_marker.pose.orientation.w = 0;
  epos_marker.scale.x = ptSize;
  epos_marker.scale.y = ptSize;
  epos_marker.scale.z = ptSize;
  epos_marker.color.a = 1.0;
  epos_marker.color.r = 0.0;
  epos_marker.color.g = 1.0;
  epos_marker.color.b = 0.0;
  epos_marker.points = epos;

  epos_marker_pub.publish(epos_marker);

  std::vector<geometry_msgs::Point> apos;
  geometry_msgs::Point apos_pt;
  apos_pt.x = actualImgTf(0,3);
  apos_pt.y = actualImgTf(1,3);
  apos_pt.z = actualImgTf(2,3);
  apos.push_back(apos_pt);
  visualization_msgs::Marker apos_marker;
 
  apos_marker.header.frame_id = "/markers";
  apos_marker.header.stamp = ros::Time();
  apos_marker.ns = "map_localize";
  apos_marker.id = 901;
  apos_marker.type = visualization_msgs::Marker::POINTS;
  apos_marker.action = visualization_msgs::Marker::ADD;
  apos_marker.pose.position.x = 0;
  apos_marker.pose.position.y = 0;
  apos_marker.pose.position.z = 0;
  apos_marker.pose.orientation.x = 0;
  apos_marker.pose.orientation.y = 0;
  apos_marker.pose.orientation.z = 0;
  apos_marker.pose.orientation.w = 0;
  apos_marker.scale.x = ptSize;
  apos_marker.scale.y = ptSize;
  apos_marker.scale.z = ptSize;
  apos_marker.color.a = 1.0;
  apos_marker.color.r = 0.0;
  apos_marker.color.g = 0.0;
  apos_marker.color.b = 1.0;
  apos_marker.points = apos;

  apos_marker_pub.publish(apos_marker);
  
  std::vector<geometry_msgs::Point> pathPts;
  for(unsigned int i = 0; i < positionList.size(); i++)
  {
    geometry_msgs::Point pt;
    pt.x = positionList[i](0);//0.5;
    pt.y = positionList[i](1);//0.5;
    pt.z = positionList[i](2);//0.5;
    pathPts.push_back(pt);
  }

  visualization_msgs::Marker path_viz;
  path_viz.header.frame_id = "/markers";
  path_viz.header.stamp = ros::Time();
  path_viz.ns = "map_localize";
  path_viz.id = 902;
  path_viz.type = visualization_msgs::Marker::LINE_STRIP;
  path_viz.action = visualization_msgs::Marker::ADD;
  path_viz.pose.position.x = 0;
  path_viz.pose.position.y = 0;
  path_viz.pose.position.z = 0;
  path_viz.pose.orientation.x = 0;
  path_viz.pose.orientation.y = 0;
  path_viz.pose.orientation.z = 0;
  path_viz.pose.orientation.w = 0;
  path_viz.scale.x = .05;
  path_viz.color.a = 1.0;
  path_viz.color.r = 0;
  path_viz.color.g = 1.0;
  path_viz.color.b = 0;
  path_viz.points = pathPts;

  path_marker_pub.publish(path_viz);


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

Mat MapLocalizer::GenerateVirtualImage(Eigen::Matrix4f tf, Eigen::Matrix3f K, int height, int width, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud, Mat& depths, Mat& mask)
{
  double scale = 1.0;
  height *= scale;
  width *= scale;

  Mat img(height, width, CV_8U, Scalar(0));
  depths = Mat(height, width, CV_32F, Scalar(-1));
  mask = Mat(height, width, CV_8U, Scalar(0));

  Eigen::MatrixXf P(3,4);
  P = K*tf.inverse().block<3,4>(0,0);
  for(unsigned int j = 0; j < cloud->points.size(); j++)
  {
    Eigen::Matrix3f Rinv = tf.inverse().block<3,3>(0,0);
    Eigen::Vector3f normal(cloud->points[j].normal_x, cloud->points[j].normal_y, cloud->points[j].normal_z);
    normal = Rinv*normal;

    Eigen::Vector4f hpt(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z, 1);
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
      img.at<uchar>(dy_idx, dx_idx) = (*reinterpret_cast<int*>(&(cloud->points[j].rgb)) & 0x0000ff);
    }
  }

  //pyrUp(img, img);//, Size(oldheight, oldwidth));
  medianBlur(img, img, 3);
  medianBlur(depths, depths, 5);
  //medianBlur(mask, mask, 3);
  return img;
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


  double* reprojError = new double;
  // Use ratio test to find good keypoint matches
  for(unsigned int j = 0; j < matches.size(); j++)
  {
    allMatches.push_back(matches[j][0]);
    if(matches[j][0].distance < matchRatio*matches[j][1].distance)
    {
      Point2f pt1 = kfc1->GetKeypoints()[matches[j][0].queryIdx].pt;
      Point2f pt2 = kfc2->GetKeypoints()[matches[j][0].trainIdx].pt;
      Mat_<double> triPt = LinearLSTriangulation(Point3d(pt1.x, pt1.y, 1), Kcv*P, Point3d(pt2.x, pt2.y, 1), Kcv*P1, reprojError);
      //std::cout << "Reproj Error: " << *reprojError << std::endl;

      if(*reprojError < 1.)
      {
        pc.push_back(pcl::PointXYZ(triPt(0), triPt(1), triPt(2)));
      
        goodMatches.push_back(matches[j][0]);
        matchKps1.push_back(kfc1->GetKeypoints()[matches[j][0].queryIdx]);
        matchKps2.push_back(kfc2->GetKeypoints()[matches[j][0].trainIdx]);
      }
    }
  }
  
  delete reprojError;
 
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

Eigen::Matrix4f MapLocalizer::StringToMatrix4f(std::string str)
{
  Eigen::Matrix4f mat;
  std::vector<double> fields;
  size_t cur_pos=0;
  size_t found_pos=0;
  size_t last_found_pos=0;

  while((found_pos = str.find(' ', cur_pos)) != std::string::npos)
  {
    fields.push_back(atof(str.substr(cur_pos, found_pos-cur_pos).c_str()));
    cur_pos = found_pos+1;
    last_found_pos = found_pos;
  }
  fields.push_back(atof(str.substr(last_found_pos).c_str()));
  if(fields.size() != 16)
  {
    std::cout << "String is not correctly formatted" << std::endl;
    return mat;
  }

  for(int i = 0; i < 16; i++)
  {
    mat(i/4,i%4) = fields[i];
  }
  return mat;
}

bool MapLocalizer::LoadPhotoscanFile(std::string filename)
{
  //FileStorage fs;
  TiXmlDocument doc(filename);

  ROS_INFO("Loading %s...", filename.c_str());
  if(!doc.LoadFile())
  {  
    ROS_ERROR("Failed to load photoscan file");
    return false;
  }
  ROS_INFO("Successfully loaded photoscan file");
  

  ROS_INFO("Loading images...");
  TiXmlHandle docHandle(&doc);
  for(TiXmlElement* chunk = docHandle.FirstChild( "document" ).FirstChild( "chunk" ).ToElement();
    chunk != NULL; chunk = chunk->NextSiblingElement("chunk"))
  {
    if (std::string(chunk->Attribute("active")) == "true")
    {
      TiXmlHandle chunkHandle(chunk);
      for(TiXmlElement* camera = chunkHandle.FirstChild("cameras").FirstChild("camera").ToElement();
        camera != NULL; camera = camera->NextSiblingElement("camera"))
      {
        std::string filename = camera->FirstChild("frames")->FirstChild("frame")->FirstChild("image")->ToElement()->Attribute("path");
        TiXmlNode* tfNode = camera->FirstChild("transform");
        if(!tfNode)
          continue;
        std::string tfStr = tfNode->ToElement()->GetText();

        //std::cout << "Loading: " << filename << std::endl;
        //std::cout << tfStr << std::endl;

        Mat img_in = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
        
        if(! img_in.data )                             
        {
          ROS_ERROR("Could not open or find the image %s", filename.c_str());
          return false;
        }

        Mat img_undistort;
        undistort(img_in, img_undistort, map_Kcv, map_distcoeffcv);
	img_in.release();

        // downsample large images to save space
        if(img_undistort.rows > 480)
        {
          double scale = 1.2*480./img_undistort.rows;
          resize(img_undistort, img_undistort, Size(0,0), scale, scale);
        }

        Eigen::Matrix4f tf = StringToMatrix4f(tfStr);
        //std::cout << tf << std::endl;
        cameras.push_back(new CameraContainer(img_undistort, tf, map_K));
        //std::cout << cameras.size() << std::endl;
      }
      //std::cout << "Found chunk " << chunk->Attribute("label") << std::endl;
    }
  }
  return true;
}


bool MapLocalizer::FindImageTfVirtualPnp(KeyframeContainer* kfc, Eigen::Matrix4f vimgTf, Eigen::Matrix3f vimgK, Eigen::Matrix4f& tf, std::string vdesc_type)
{
  tf = Eigen::MatrixXf::Identity(4,4);

  // Get virtual image and depth map
  Mat depth, mask;
  Mat vimg;
  ros::Time start = ros::Time::now();
  if(virtual_image_source == "gazebo")
  {
    vimg = GetVirtualImageFromTopic(depth, mask);
  }
  else if(virtual_image_source == "point_cloud")
  {
    vimg = GenerateVirtualImage(vimgTf, vimgK, kfc->GetImage().rows, kfc->GetImage().cols, map_cloud, depth, mask);
  }
  else
  {
    ROS_ERROR("Invalid virtual_image_source");
    return false;
  }
  ROS_INFO("VirtualPnP: generate virtual img time: %f", (ros::Time::now()-start).toSec());
  
#if 0
  Mat depth_im;
  double min_depth, max_depth;
  minMaxLoc(depth, &min_depth, &max_depth);    
  depth.convertTo(depth_im, CV_8U, 255.0/(max_depth-min_depth), 0);//-255.0/min_depth);

  namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Query", kfc->GetImage() ); 
  namedWindow( "Virtual", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Virtual", vimg ); 
  namedWindow( "Depth", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Depth", depth_im ); 
  namedWindow( "Mask", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Mask", mask ); 
  waitKey(1);
#endif

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
    ORB orb(1000);
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
    imshow("matches", img_matches);
    waitKey(0);
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
  if(!RansacPnP(matchPts3d, matchPts, vimgTf.inverse(), tfran))
  {
    return false;
  }

  tf = tfran.inverse();
  return true;
}

bool MapLocalizer::RansacPnP(const std::vector<Point3f>& matchPts3d, const std::vector<Point2f>& matchPts, Eigen::Matrix4f tfguess, Eigen::Matrix4f& tf)
{
  Mat distcoeffcvPnp = (Mat_<double>(4,1) << 0, 0, 0, 0);
  tf = Eigen::MatrixXf::Identity(4,4);
  Mat Rvec, t;
  Mat Rguess = (Mat_<double>(3,3) << tfguess(0,0), tfguess(0,1), tfguess(0,2),
                                     tfguess(1,0), tfguess(1,1), tfguess(1,2),
                                     tfguess(2,0), tfguess(2,1), tfguess(2,2));
  Rodrigues(Rguess, Rvec);
  t = (Mat_<double>(3,1) << tfguess(0,3), tfguess(1,3), tfguess(2,3));
  
  // RANSAC PnP
  const int niter = 50;//50; // Assumes about 45% outliers
  const double reprojThresh = 5.0; // in pixels
  const int m = 4; // points per sample
  const int inlier_ratio_cutoff = 0.4; 
  std::vector<int> ind;
  for(unsigned int i = 0; i < matchPts.size(); i++)
  {
    ind.push_back(i);
  }

  std::vector<int> bestInliersIdx;
  bool abort = false;
  ros::Time start = ros::Time::now();
  //#pragma omp parallel for
  for(int i = 0; i < niter; i++)
  {
    //#pragma omp flush (abort)
    //if(abort)
    //{
    //  continue;
    //}

    Eigen::Matrix4f rand_tf;
    // Get m random points
    std::random_shuffle(ind.begin(), ind.end());
    std::vector<int> randInd(ind.begin(), ind.begin()+m);

    std::vector<Point3f> rand_matchPts3d;
    std::vector<Point2f> rand_matchPts;
    for(int j = 0; j < m; j++)
    {
      rand_matchPts3d.push_back(matchPts3d[randInd[j]]);
      rand_matchPts.push_back(matchPts[randInd[j]]);
    }

    Mat ran_Rvec, ran_t;
    Rvec.copyTo(ran_Rvec);
    t.copyTo(ran_t);

    solvePnP(rand_matchPts3d, rand_matchPts, Kcv, distcoeffcvPnp, ran_Rvec, ran_t, true, CV_P3P);

    // Test for inliers
    std::vector<Point2f> reprojPts;
    projectPoints(matchPts3d, ran_Rvec, ran_t, Kcv, distcoeffcvPnp, reprojPts);
    std::vector<int> inliersIdx;
    for(unsigned int j = 0; j < reprojPts.size(); j++)
    {
      double reprojError = sqrt((reprojPts[j].x-matchPts[j].x)*(reprojPts[j].x-matchPts[j].x) + (reprojPts[j].y-matchPts[j].y)*(reprojPts[j].y-matchPts[j].y));

      if(reprojError < reprojThresh)
      {
        inliersIdx.push_back(j);
      }
    }

    //#pragma omp critical
    {
      if(inliersIdx.size() > bestInliersIdx.size())
      {
        bestInliersIdx = inliersIdx;
      } 
      if(bestInliersIdx.size() > inlier_ratio_cutoff*matchPts.size())
      {
        //std::cout << "Pnp abort n=" << i << std::endl;
        //abort = true;  
        //#pragma omp flush (abort)
      }
    }
    
  } 
  std::cout << "Num inliers: " << bestInliersIdx.size() << "/" << matchPts.size() << std::endl;
  if(bestInliersIdx.size() < 10)
  {
    ROS_WARN("ransacPnP: Could not find enough inliers");
    return false;
  }  
  ROS_INFO("PnpRansac: Ransac time: %f", (ros::Time::now()-start).toSec());  

  std::vector<Point3f> inlierPts3d;
  std::vector<Point2f> inlierPts2d;
  for(unsigned int i = 0; i < bestInliersIdx.size(); i++)
  {
    inlierPts3d.push_back(matchPts3d[bestInliersIdx[i]]);
    inlierPts2d.push_back(matchPts[bestInliersIdx[i]]);
  }

  start = ros::Time::now();
  solvePnP(inlierPts3d, inlierPts2d, Kcv, distcoeffcvPnp, Rvec, t, true);
  ROS_INFO("PnpRansac: Pnp Inliers time: %f", (ros::Time::now()-start).toSec());  
    
  Mat R;
  Rodrigues(Rvec, R);

  tf << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2),
             0,      0,      0,    1;
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


void MapLocalizer::TestFindImageTfSfm()
{
/*
  int l,w,h;
  l = w = h = 5;
  int numPts = 100;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.resize(numPts);
  
  for(int i = 0; i < numPts; i++)
  {
    cloud->points[i].x = l * rand()/(RAND_MAX + 1.0);
    cloud->points[i].y = w * rand()/(RAND_MAX + 1.0);
    cloud->points[i].z = h * rand()/(RAND_MAX + 1.0);
  }

  double t=M_PI/6;
  Eigen::Matrix4f tf1, tf2;
  tf1 << cos(t),-sin(t),  0,   1,
         sin(t), cos(t),  0, 0.5,
              0,      0,  1,   0,
              0,      0,  0,   1;
  tf2 = AngleAxis<float>(M_PI/12, Vector3f::UnitZ())*Translation<float,3>(0.5, 0.5, 0.5)*tf1;

  Eigen::MatrixXf P1(3,4), P2(3,4);
  P1 = K*tf1.inverse().block<3,4>(0,0);
  P2 = K*tf2.inverse().block<3,4>(0,0);

  int width, height;
  width = 752;
  height = 480;
  std::vector<KeyPoint> kp1, kp2;
  std::vector<DMatch> ptmatches;
  for(unsigned int j = 0; j < cloud->points.size(); j++)
  {
    Eigen::Matrix3f Rinv1 = tf1.inverse().block<3,3>(0,0);
    Eigen::Matrix3f Rinv2 = tf2.inverse().block<3,3>(0,0);

    Eigen::Vector4f hpt(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z, 1);
    Eigen::Vector3f impt1 = P1*hpt;
    Eigen::Vector3f impt2 = P2*hpt;
    impt1 /= impt1(2);
    impt2 /= impt2(2);
    double dx_idx1 = impt1(0);
    double dy_idx1 = impt1(1);
    double dx_idx2 = impt2(0);
    double dy_idx2 = impt2(1);
    if(dx_idx1 < 0  || dx_idx1 >= width || dy_idx1 < 0 || dy_idx1 >= height)
    {
      continue;
    }
    if(dx_idx2 < 0  || dx_idx2 >= width || dy_idx2 < 0 || dy_idx2 >= height)
    {
      continue;
    }
    double depth1 = (tf1.inverse()*hpt)(2);
    double depth2 = (tf2.inverse()*hpt)(2);
    
    if(depth1 > 0 && depth2 > 0)
    {
      kp1.push_back(KeyPoint(dx_idx1, dy_idx1, 1));  
      kp2.push_back(KeyPoint(dx_idx2, dy_idx2, 1));  
      ptmatches.push_back(DMatch(kp1.size()-1, kp2.size()-1, 0));
    }
  }

  Matx34d P(1,0,0,0,
            0,1,0,0, 
            0,0,1,0);
  Matx34d P1(1,0,0,0,
            0,1,0,0,
            0,0,1,0);

  std::vector<KeyPoint> matchPts1_good;
  std::vector<KeyPoint> matchPts2_good;
  std::vector<CloudPoint> outCloud;
  double reproj_error;

  bool goodF = FindCameraMatrices(Kcv, Kcv.inv(), distcoeffcv, kp1, kp2, matchPts1_good, matchPts2_good, P, P1, ptmatches, outCloud, reproj_error);

  std::cout << tf1 << std::endl << std::endl << tf2 << std::endl;
  std::cout << Mat(P1) << std::endl;
*/
}

Eigen::Matrix4f MapLocalizer::FindImageTfSfm(KeyframeContainer* img, std::vector< KeyframeMatch > matches, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs)
{
  Eigen::Matrix4f tf = Eigen::MatrixXf::Identity(4,4);
  std::vector<Eigen::Matrix4f> goodPs;
  goodMatches.clear();
  goodTVecs.clear();

  bool useH = false;
  std::vector<double> reproj_errors;

  for(unsigned int i = 0; i < matches.size(); i++)
  {
    Matx34d P(1,0,0,0,
              0,1,0,0, 
              0,0,1,0);
    Matx34d P1(1,0,0,0,
              0,1,0,0,
              0,0,1,0);
    Matx34d P2(1,0,0,0,
              0,1,0,0,
              0,0,1,0);

    std::vector<KeyPoint> matchPts1_good;
    std::vector<KeyPoint> matchPts2_good;
    std::vector<DMatch> ptmatches = matches[i].matches; // Uses only the good matches found with ratio test
    //std::vector<DMatch> ptmatches = matches[i].allMatches; // Uses all keypoint matches
    std::vector<CloudPoint> outCloud;
    double reproj_error;

    bool goodF = FindCameraMatrices(Kcv, Kcv.inv(), distcoeffcv, img->GetKeypoints(), matches[i].kfc->GetKeypoints(), matchPts1_good, matchPts2_good, P, P1, ptmatches, outCloud, reproj_error);
    std::cout << "goodF: " << goodF << std::endl; 
    
    if(goodF)
    {
      Eigen::Matrix4f P1m;
      P1m <<   P1(0,0), P1(0,1), P1(0,2), P1(0,3),
               P1(1,0), P1(1,1), P1(1,2), P1(1,3),
               P1(2,0), P1(2,1), P1(2,2), P1(2,3),
                     0,       0,       0,       1;
      reproj_errors.push_back(reproj_error);
      goodPs.push_back(P1m);
      goodMatches.push_back(matches[i]);
    } 
  }
  if(goodPs.size() < 2)
  {
    ROS_WARN("Not enough good matches.  Found %d. Trying H instead.", (int)goodPs.size());
    useH = true;
  } 

  if(useH)
  {
    goodPs.clear();
    goodMatches.clear();
    for(unsigned int i = 0; i < matches.size(); i++)
    {
      Matx34d P(1,0,0,0,
                0,1,0,0, 
                0,0,1,0);
      Matx34d P1(1,0,0,0,
                0,1,0,0,
                0,0,1,0);
      Matx34d P2(1,0,0,0,
                0,1,0,0,
                0,0,1,0);

      std::vector<KeyPoint> matchPts1_good;
      std::vector<KeyPoint> matchPts2_good;
      std::vector<DMatch> ptmatches = matches[i].matches; // Uses only the good matches found with ratio test
      //std::vector<DMatch> ptmatches = matches[i].allMatches; // Uses all keypoint matches
      std::vector<CloudPoint> outCloud;
  
      bool goodH = FindCameraMatricesWithH(Kcv, Kcv.inv(), distcoeffcv, img->GetKeypoints(), matches[i].kfc->GetKeypoints(), matchPts1_good, matchPts2_good, P1, ptmatches);
      std::cout << "goodH: " << goodH << std::endl; 
    
      if(goodH)
      {
        Eigen::Matrix4f P1m;
        P1m <<   P1(0,0), P1(0,1), P1(0,2), P1(0,3),
                 P1(1,0), P1(1,1), P1(1,2), P1(1,3),
                 P1(2,0), P1(2,1), P1(2,2), P1(2,3),
                       0,       0,       0,       1;
        goodPs.push_back(P1m);
        goodMatches.push_back(matches[i]);
      } 
    }
  }

  if(goodPs.size() < 2)
  {
    ROS_WARN("Not enough good matches.  Found %d.", (int)goodPs.size());
    return tf;
  }
 
  std::cout << "--------------------" << std::endl; 
  std::cout << "# of good matches: " << goodPs.size() << "/" << matches.size() << std::endl; 
  std::cout << "Actual Tf: " << std::endl << img->GetTf() << std::endl;
  Eigen::MatrixXf lsA(3*goodPs.size(), 3);
  Eigen::VectorXf lsb(3*goodPs.size());
  Eigen::VectorXf Wvec(3*goodPs.size());
  for(unsigned int i = 0; i < goodPs.size(); i++)
  {
    //Eigen::Matrix4f tf1to2 = Eigen::MatrixXf::Identity(4,4);
    //tf1to2.block<3,3>(0,0) = goodPs[i].block<3,3>(0,0);
    //tf1to2.block<3,1>(0,3) = -goodPs[i].block<3,3>(0,0)*goodPs[i].block<3,1>(0,3);
    //Eigen::Matrix4f imgWorldtf = goodMatches[i].kfc->GetTf()*tf1to2;//*goodPs[i]; //Image World tf
    Eigen::Matrix4f imgWorldtf = goodMatches[i].kfc->GetTf()*goodPs[i]; //Image World tf
    
    Eigen::Vector3f a = goodMatches[i].kfc->GetTf().block<3,1>(0,3); // tf of match
    Eigen::Vector3f d = imgWorldtf.block<3,1>(0,3) - a; // direction from match to image in world frame
    d = d/d.norm();
    goodTVecs.push_back(d);

    for(int j = 0; j < 3; j++)
    {
      if(!useH)
      {
        Wvec(3*i+j) = sqrt(1./reproj_errors[i]);
      }
      lsb(i*3+j) = 0;
      for(int k = 0; k < 3; k++) 
      {
        if(j == k)
        {
          lsA(i*3+j,k) = 1-d(j)*d(j);
          lsb(i*3+j) += a(j) - a(j)*d(j)*d(j);
        }
        else
        {
          lsA(i*3+j,k) = -d(j)*d(k);
          lsb(i*3+j) += -a(k)*d(k)*d(j);
        }
      }
    }
    
    //std::cout << "--------------------" << std::endl; 
    //std::cout << imgWorldtf << std::endl << std::endl;
    //std::cout << goodMatches[i].kfc->GetTf().transpose()*goodPs[i] << std::endl << std::endl;
  }
  // Weight LS by the inverse reprojection error if it is available
  if(!useH)
  {
    lsA = Wvec.asDiagonal()*lsA;
    lsb = Wvec.asDiagonal()*lsb;
  }
  Eigen::VectorXf lsTrans = lsA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(lsb);
  
  std::cout << "--------------------" << std::endl; 
  std::cout << "Least squares T:" << std::endl << lsTrans << std::endl; 
  tf.block<3,1>(0,3) = lsTrans;

  // Rotation averaging (5.3) http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
  /*
  Eigen::Matrix4f imgWorldtf = goodMatches[0].kfc->GetTf()*goodPs[0];
  Eigen::Matrix3f R = imgWorldtf.block<3,3>(0,0);
  float eps = .001;
  while(1)
  {
    Eigen::MatrixXf r = Eigen::MatrixXf::Zero(3,3);
    for(int i = 0; i < goodPs.size(); i++)
    {
      Eigen::MatrixXf Ri = (goodMatches[i].kfc->GetTf()*goodPs[i]).block<3,3>(0,0);
      Eigen::Matrix3f RtRi = R.transpose()*Ri;
      r += RtRi.log()/goodPs.size();
    }
    if(r.norm() < eps)
      break;
    R = R*r.exp();
  }
  */
  double best_reproj_error = 99999;
  for(unsigned int i = 0; i < reproj_errors.size(); i++)
  {
    if(reproj_errors[i] < best_reproj_error)
    {
      tf.block<3,3>(0,0) = (goodMatches[i].kfc->GetTf()*goodPs[i].inverse()).block<3,3>(0,0);
      best_reproj_error = reproj_errors[i];
    }
  }
  return tf;
}
