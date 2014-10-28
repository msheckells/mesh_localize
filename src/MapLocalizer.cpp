#include "map_localize/MapLocalizer.h"
#include <tinyxml.h>
#include <algorithm>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 
#include <fstream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cv_bridge/cv_bridge.h>

#include "map_localize/FindCameraMatrices.h"
#include "map_localize/Triangulation.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
//#define SHOW_MATCHES_ 

MapLocalizer::MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private):
    currentKeyframe(NULL),
    isLocalized(false),
    numLocalizeRetrys(0),
    nh(nh),
    nh_private(nh_private)
{

  // TODO: read K and distcoeff from param file
  K << 701.907522339299, 0, 352.73599016194, 0, 704.43277859417, 230.636873050629, 0, 0, 1;
  distcoeff = Eigen::VectorXf(5);
  distcoeff << -0.456758192707853, 0.197636354824418, 0.000543685887014507, 0.000401738655456894, 0;

  Kcv = Matx33d(K(0,0), K(0,1), K(0,2),
              K(1,0), K(1,1), K(1,2),
              K(2,0), K(2,1), K(2,2)); 
  distcoeffcv = (Mat_<double>(5,1) << distcoeff(0), distcoeff(1), distcoeff(2), distcoeff(3), distcoeff(4)); 

  // TODO: make filepath param
  pc_filename = "bin/map.pcd";
  mesh_filename = "bin/map.stl";
  map_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  if(pcl::io::loadPCDFile<pcl::PointXYZ> (pc_filename, *map_cloud) == -1)
  {
    std::cout << "Could not open point cloud " << pc_filename << std::endl;
    return;
  }

  photoscan_filename = "/home/matt/Documents/campus_doc.xml";
  if(!LoadPhotoscanFile(photoscan_filename, "data/ASiftKeypointsAndDescriptors.bin", true))
  {
    return;
  }

  //std::cout << "Mapping features to point cloud..." << std::flush;
  //map_features = MapFeatures(keyframes, map_cloud);
  //std::cout << "done" << std::endl;

  srand(time(NULL));
  
  map_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/map", 0);
  match_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/match_points", 0);
  tvec_marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/map_localize/t_vectors", 0);
  epos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/estimated_position", 0);
  apos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/actual_position", 0);
  path_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/estimated_path", 0);
  pointcloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/map_localize/pointcloud", 0);

  image_subscriber = nh.subscribe("image", 0, &MapLocalizer::HandleImage, this, ros::TransportHints().tcpNoDelay());

  timer = nh_private.createTimer(ros::Duration(0.1), &MapLocalizer::spin, this);
}

MapLocalizer::~MapLocalizer()
{
  for(unsigned int i = 0; i < keyframes.size(); i++)
  {
    delete keyframes[i];
  }
  keyframes.clear();
}


void MapLocalizer::HandleImage(sensor_msgs::ImageConstPtr msg)
{
  if(!currentKeyframe)
  {
    ROS_INFO("Processing new image");

    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(msg);
    currentKeyframe = new KeyframeContainer(cvImg->image);
  }
}

void MapLocalizer::spin(const ros::TimerEvent& e)
{
  
  PublishMap();
  //if(currentKeyframe)
  {
    //Mat test = imread("/home/matt/uav_image_data/run11/frame0039.jpg", CV_LOAD_IMAGE_GRAYSCALE );
    //KeyframeContainer* kf = new KeyframeContainer(test, Eigen::Matrix4f());
    //KeyframeContainer* kf = currentKeyframe;
    KeyframeContainer* kf = keyframes[150];//keyframes[rand() % keyframes.size()];
    
#if 0
    Eigen::Matrix4f imgTf = FindImageTfPnp(kf, map_features);
    currentPosition = imgTf.block<3,1>(0,3);
    positionList.push_back(currentPosition);
    PublishTfViz(imgTf, kf->GetTf());
#else
    
    std::vector< KeyframeMatch > matches = FindImageMatches(kf, 5);//, isLocalized); // Only do local search if position is known
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
    std::vector<pcl::PointXYZ> pclCloud = GetPointCloudFromFrames(matches[1].kfc, matches[2].kfc);
    std::vector<Point3d> cvCloud = PCLToPoint3d(pclCloud);
    std::vector<int> planeIdx, nonplaneIdx;
    //ROS_INFO("Got point cloud");
    if(pclCloud.size() > 5)
    {
      std::vector<int> inliers = FindPlaneInPointCloud(pclCloud);
      std::vector<pcl::PointXYZ> planeCloud;
      for(unsigned int i = 0; i < inliers.size(); i++)
      {
        planeCloud.push_back(pclCloud[inliers[i]]);
      }
      PublishPointCloud(map_cloud);
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
#ifdef SHOW_MATCHES_
    namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Query", kf->GetImage() ); 
    waitKey(0);
    for(int i = 0; i < matches.size(); i++)
    {
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
    // For now just delete currentKeyframe, should probably add to keyframe list 
    // if tf estimate is good enough
    //delete currentKeyframe;
    //currentKeyframe = NULL;
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
  pc->header.frame_id = "/markers";
  pc->header.stamp = ros::Time();
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
  pcl::PointCloud<pcl::PointXYZ> msg;
  msg.header.frame_id = "/markers";
  msg.header.stamp = ros::Time();

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
  marker.header.frame_id = "/markers";
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
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.a = 0.6;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  //only if using a MESH_RESOURCE marker type:
  marker.mesh_resource = std::string("package://map_localize/") + mesh_filename;

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
 
#ifdef SHOW_MATCHES_
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

bool MapLocalizer::LoadPhotoscanFile(std::string filename, std::string desc_filename, bool load_descs)
{
  //FileStorage fs;
  std::ifstream desc_file;
  TiXmlDocument doc(filename);

  std::cout << "Loading " << filename << "..." << std::flush;
  if(!doc.LoadFile())
  {  
    std::cout << "failed" << std::endl;
    return false;
  }
  std::cout << "done" << std::endl;
  
  if(load_descs)
  {
    std::cout << "Loading keypoints and descriptors..." << std::flush;
    //fs = FileStorage(desc_filename, FileStorage::READ);
    desc_file.open(desc_filename.c_str(), std::ifstream::binary);
    //if(!fs.isOpened())
    if(!desc_file.is_open())
    {
      std::cout << "Could not open descriptor file " << desc_filename << std::endl;
      return false;
    }
    std::cout << "done" << std::endl;
  }

  std::cout << "Loading images..." << std::flush;
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
        std::string filename = std::string("/home/matt/Documents/") + camera->FirstChild("frames")->FirstChild("frame")->FirstChild("image")->ToElement()->Attribute("path");
        TiXmlNode* tfNode = camera->FirstChild("transform");
        if(!tfNode)
          continue;
        std::string tfStr = tfNode->ToElement()->GetText();

        //std::cout << "Loading: " << filename << std::endl;
        //std::cout << tfStr << std::endl;

        Mat img_in = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
        
        if(! img_in.data )                             
        {
          std::cout <<  "Could not open or find the image " << filename << std::endl ;
          return false;
        }

        Mat img_undistort;
        undistort(img_in, img_undistort, Kcv, distcoeffcv);
	img_in.release();

        Eigen::Matrix4f tf = StringToMatrix4f(tfStr);
        //std::cout << tf << std::endl;

        KeyframeContainer* kfc;
        if(load_descs && desc_filename != "")
        {
          /*std::stringstream ss;
          ss << keyframes.size();
          
          Mat descriptors;
          std::vector<KeyPoint> keypoints;
          
          fs[std::string("descriptors") + ss.str()] >> descriptors;
          FileNode kpn = fs[std::string("keypoints") + ss.str()];
          read(kpn, keypoints); 
          */
          std::vector<KeyPoint> keypoints;
          int size;
          desc_file.read((char*)&size, sizeof(int));
          for(int i = 0; i < size; i++)
          {
            KeyPoint kp;
            desc_file.read((char*)&kp, sizeof(KeyPoint));
            keypoints.push_back(kp);
          }
          int rows;
          int cols;
          int elemSize;
          desc_file.read((char*)&rows, sizeof(int));
          desc_file.read((char*)&cols, sizeof(int));
          desc_file.read((char*)&elemSize, sizeof(int));
          
          unsigned char* data = new unsigned char[rows*cols*elemSize];
	  desc_file.read((char*)data, sizeof(unsigned char)*rows*cols*elemSize);
          
          Mat descriptors(Size(cols,rows), CV_32F, data);
           
          kfc = new KeyframeContainer(img_undistort, tf, K, keypoints, descriptors);
        }
        else
        {  
          kfc = new KeyframeContainer(img_undistort, tf, K);
        }
        keyframes.push_back(kfc);    
        //std::cout << keyframes.size() << std::endl;
      }
      //std::cout << "Found chunk " << chunk->Attribute("label") << std::endl;
    }
  }
  std::cout << "done" << std::endl;

  if(load_descs)
  {
    std::cout << "File storage released" << std::endl;
    //fs.release();
    desc_file.close();
  }
  else if(desc_filename != "")
  {
    WriteDescriptorsToFile(desc_filename);
  }
  return true;
}

bool MapLocalizer::WriteDescriptorsToFile(std::string filename)
{
  std::ofstream file;
  file.open(filename.c_str(), std::ios::out | std::ios::binary);

  if(!file.is_open())
  {
    std::cout << "Could not open descriptor file " << filename << std::endl;
    return false;
  }

  std::cout << "Saving keypoints and descriptors..." << std::flush;
  for(unsigned int i = 0; i < keyframes.size(); i++)
  {
    int size = keyframes[i]->GetKeypoints().size();
    file.write((char*)&size, sizeof(int));
    for(int j = 0; j < size; j++)
    {
      KeyPoint kp = keyframes[i]->GetKeypoints()[j];
      file.write((char*)&kp, sizeof(KeyPoint));
    }
    int rows = keyframes[i]->GetDescriptors().rows;
    int cols = keyframes[i]->GetDescriptors().cols;
    int elemSize = keyframes[i]->GetDescriptors().elemSize1();
    unsigned char* data = keyframes[i]->GetDescriptors().data;
   
    file.write((char*)&rows, sizeof(int));
    file.write((char*)&cols, sizeof(int));
    file.write((char*)&elemSize, sizeof(int));
    file.write((char*)data, sizeof(unsigned char)*rows*cols*elemSize);
  }

  file.close();
  std::cout << "done" << std::endl;
  return true;
  /*
  FileStorage fs(filename, FileStorage::WRITE);
  
  if(!fs.isOpened())
  {
    return false;
  }

  for(unsigned int i = 0; i < keyframes.size(); i++)
  {
    std::stringstream ss;
    ss << i;
    write(fs, std::string("keypoints") + ss.str(), keyframes[i]->GetKeypoints());
    write(fs, std::string("descriptors") + ss.str(), keyframes[i]->GetDescriptors());
  }
  fs.release();
  */
  
  return true;
}

Eigen::Matrix4f MapLocalizer::FindImageTfPnp(KeyframeContainer* kfc, const MapFeatures& mf)
{
  Eigen::Matrix4f tf;
  
  // Find image features matches in map
  const double matchRatio = 0.8;

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
  return tf;
}

std::vector< KeyframeMatch > MapLocalizer::FindImageMatches(KeyframeContainer* img, int k, bool usePos)
{
  const double numMatchThresh = 0;//0.16;
  const double matchRatio = 0.9;
  std::vector< KeyframeMatch > kfMatches;
  unsigned int searchBound = keyframes.size();

  if(usePos)
  {
    ROS_INFO("Peforming local search");
    searchBound = 5*k*pow(1.8, numLocalizeRetrys);
    if(searchBound >= keyframes.size())
    {
      searchBound = keyframes.size();
    }
    else 
    {
      KeyframePositionSorter kps(this);
      std::sort(keyframes.begin(), keyframes.end(), kps);
    }
  }

  // Find potential frame matches
  for(unsigned int i = 0; i < searchBound; i++)
  {
    //std::cout << i/double(keyframes.size()) << std::endl;

    FlannBasedMatcher matcher;
    std::vector < std::vector< DMatch > > matches;
    matcher.knnMatch( img->GetDescriptors(), keyframes[i]->GetDescriptors(), matches, 2 );

    std::vector< DMatch > goodMatches;
    std::vector< DMatch > allMatches;
    std::vector<Point2f> matchPts1;
    std::vector<Point2f> matchPts2;
    std::vector<KeyPoint> matchKps1;
    std::vector<KeyPoint> matchKps2;
    
    // Use ratio test to find good keypoint matches
    for(unsigned int j = 0; j < matches.size(); j++)
    {
      allMatches.push_back(matches[j][0]);
      if(matches[j][0].distance < matchRatio*matches[j][1].distance)
      {
        goodMatches.push_back(matches[j][0]);
        matchPts1.push_back(img->GetKeypoints()[matches[j][0].queryIdx].pt);
        matchPts2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx].pt);
        matchKps1.push_back(img->GetKeypoints()[matches[j][0].queryIdx]);
        matchKps2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx]);
      }
    }
    if(goodMatches.size() >= numMatchThresh*matches.size())
    {
      //std:: cout << "Found Match!" << std::endl;
      kfMatches.push_back(KeyframeMatch(keyframes[i], goodMatches, allMatches, matchPts1, matchPts2, matchKps1, matchKps2));
    }
  }

  k = (kfMatches.size() < k) ? kfMatches.size() : k;
  std::sort(kfMatches.begin(), kfMatches.end());

  return std::vector< KeyframeMatch > (kfMatches.begin(), kfMatches.begin()+k);
}

Eigen::Matrix4f MapLocalizer::FindImageTfSfm(KeyframeContainer* img, std::vector< KeyframeMatch > matches, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs)
{
  Eigen::Matrix4f tf = Eigen::MatrixXf::Identity(4,4);
  std::vector<Eigen::Matrix4f> goodPs;
  goodMatches.clear();
  goodTVecs.clear();

  bool useH = false;

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

    bool goodF = FindCameraMatrices(Kcv, Kcv.inv(), distcoeffcv, img->GetKeypoints(), matches[i].kfc->GetKeypoints(), matchPts1_good, matchPts2_good, P, P1, ptmatches, outCloud);
    std::cout << "goodF: " << goodF << std::endl; 
    
    if(goodF)
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
  tf.block<3,3>(0,0) = R;
  */
  return tf;
}
