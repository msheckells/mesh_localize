#include "map_localize/MapLocalizer.h"
#include <tinyxml.h>
#include <algorithm>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 
#include <Eigen/Dense>

#include "map_localize/FindCameraMatrices.h"

#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

MapLocalizer::MapLocalizer(ros::NodeHandle nh, ros::NodeHandle nh_private):
    nh(nh),
    nh_private(nh_private)
{

  // TODO: make filepath param
  std::string filename = "/home/matt/Documents/doc.xml";
  if(!LoadPhotoscanFile(filename, "data/KeypointsAndDescriptors.yml", true))
  {
    return;
  }

  /* Test Matches */
  srand(time(NULL));
/*
  namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Query", kf->GetImage() ); 
  waitKey(0);
  for(int i = 0; i < goodMatches.size(); i++)
  {
    namedWindow( "Match", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Match", goodMatches[i].kfc->GetImage() ); 
    waitKey(0);
  }  
*/
  /****/  
  map_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/map", 0);
  match_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/match_points", 0);
  tvec_marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/map_localize/t_vectors", 0);
  epos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/estimated_position", 0);
  apos_marker_pub = nh.advertise<visualization_msgs::Marker>("/map_localize/actual_position", 0);

  image_subscriber = nh.subscribe("image", 1, &MapLocalizer::HandleImage, this, ros::TransportHints().tcpNoDelay());

  timer = nh_private.createTimer(100.0, &MapLocalizer::spin, this);
}

MapLocalizer::~MapLocalizer()
{
  for(int i = 0; i < keyframes.size(); i++)
  {
    delete keyframes[i];
  }
  keyframes.clear();
}


void MapLocalizer::HandleImage(sensor_msgs::Image msg)
{

}

void MapLocalizer::spin(const ros::TimerEvent& e)
{
  //Mat test = imread("/home/matt/uav_image_data/run9/frame0332.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  //KeyframeContainer* kf = new KeyframeContainer(test, Eigen::Matrix4f());
  KeyframeContainer* kf = keyframes[150];//keyframes[rand() % keyframes.size()];
  std::vector< KeyframeMatch > matches = FindImageMatches(kf, 10);
  std::vector< KeyframeMatch > goodMatches;
  std::vector< Eigen::Vector3f > goodTVecs;

  Eigen::Matrix4f imgTf = FindImageTf(kf, matches, goodMatches, goodTVecs);

  if(goodMatches.size() < 2)
    return;
 
  PublishTfViz(imgTf, kf->GetTf(), goodMatches, goodTVecs);
}

void MapLocalizer::PublishTfViz(Eigen::Matrix4f imgTf, Eigen::Matrix4f actualImgTf, std::vector< KeyframeMatch > matches, std::vector< Eigen::Vector3f > tvecs)
{
  const double ptSize = 0.1;
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
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1.0;
  marker.scale.y = 1.0;
  marker.scale.z = 1.0;
  marker.color.a = 1.0;
  marker.color.r = 0.5;
  marker.color.g = 0.5;
  marker.color.b = 0.5;
  //only if using a MESH_RESOURCE marker type:
  marker.mesh_resource = std::string("package://map_localize/") + std::string("bin/map.stl");

  map_marker_pub.publish(marker);

  std::vector<geometry_msgs::Point> match_pos;
  for(int i = 0; i < matches.size(); i++)
  {
    geometry_msgs::Point pt;
    pt.x = matches[i].kfc->GetTf()(0,3);
    pt.y = matches[i].kfc->GetTf()(1,3);
    pt.z = matches[i].kfc->GetTf()(2,3);
    match_pos.push_back(pt);
  }
  visualization_msgs::Marker match_marker;

  match_marker.header.frame_id = "/world";
  match_marker.header.stamp = ros::Time();
  match_marker.ns = "map_localize";
  match_marker.id = 1;
  match_marker.type = visualization_msgs::Marker::POINTS;
  match_marker.action = visualization_msgs::Marker::ADD;
  match_marker.pose.position.x = 0;
  match_marker.pose.position.y = 0;
  match_marker.pose.position.z = 0;
  match_marker.pose.orientation.x = 0.0;
  match_marker.pose.orientation.y = 0.0;
  match_marker.pose.orientation.z = 0.0;
  match_marker.pose.orientation.w = 1.0;
  match_marker.scale.x = ptSize;
  match_marker.scale.y = ptSize;
  match_marker.scale.z = ptSize;
  match_marker.color.a = 1.0;
  match_marker.color.r = 1.0;
  match_marker.color.g = 0.0;
  match_marker.color.b = 0.0;
  match_marker.points = match_pos;

  match_marker_pub.publish(match_marker);

  std::vector<visualization_msgs::Marker> tvec_markers;
  for(int i = 0; i < matches.size(); i++)
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

    tvec_marker.header.frame_id = "/world";
    tvec_marker.header.stamp = ros::Time();
    tvec_marker.ns = "map_localize";
    tvec_marker.id = 2+i;
    tvec_marker.type = visualization_msgs::Marker::ARROW;
    tvec_marker.action = visualization_msgs::Marker::ADD;
    tvec_marker.pose.position.x = 0; 
    tvec_marker.pose.position.y = 0; 
    tvec_marker.pose.position.z = 0; 
    tvec_marker.pose.orientation.x = 0.0;
    tvec_marker.pose.orientation.y = 0.0;
    tvec_marker.pose.orientation.z = 0.0;
    tvec_marker.pose.orientation.w = 1.0;
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


  std::vector<geometry_msgs::Point> epos;
  geometry_msgs::Point epos_pt;
  epos_pt.x = imgTf(0,3);
  epos_pt.y = imgTf(1,3);
  epos_pt.z = imgTf(2,3);
  epos.push_back(epos_pt);
  visualization_msgs::Marker epos_marker;
 
  epos_marker.header.frame_id = "/world";
  epos_marker.header.stamp = ros::Time();
  epos_marker.ns = "map_localize";
  epos_marker.id = 900;
  epos_marker.type = visualization_msgs::Marker::POINTS;
  epos_marker.action = visualization_msgs::Marker::ADD;
  epos_marker.pose.position.x = 0;
  epos_marker.pose.position.y = 0;
  epos_marker.pose.position.z = 0;
  epos_marker.pose.orientation.x = 0.0;
  epos_marker.pose.orientation.y = 0.0;
  epos_marker.pose.orientation.z = 0.0;
  epos_marker.pose.orientation.w = 1.0;
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
 
  apos_marker.header.frame_id = "/world";
  apos_marker.header.stamp = ros::Time();
  apos_marker.ns = "map_localize";
  apos_marker.id = 901;
  apos_marker.type = visualization_msgs::Marker::POINTS;
  apos_marker.action = visualization_msgs::Marker::ADD;
  apos_marker.pose.position.x = 0;
  apos_marker.pose.position.y = 0;
  apos_marker.pose.position.z = 0;
  apos_marker.pose.orientation.x = 0.0;
  apos_marker.pose.orientation.y = 0.0;
  apos_marker.pose.orientation.z = 0.0;
  apos_marker.pose.orientation.w = 1.0;
  apos_marker.scale.x = ptSize;
  apos_marker.scale.y = ptSize;
  apos_marker.scale.z = ptSize;
  apos_marker.color.a = 1.0;
  apos_marker.color.r = 0.0;
  apos_marker.color.g = 0.0;
  apos_marker.color.b = 1.0;
  apos_marker.points = apos;

  apos_marker_pub.publish(apos_marker);
  
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
  tf::Quaternion q;
  q.setRPY(0.0, 0, 0);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "map_localize"));
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
  FileStorage fs;
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
    fs = FileStorage(desc_filename, FileStorage::READ);
    if(!fs.isOpened())
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

        Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE );
        if(! img.data )                             
        {
          std::cout <<  "Could not open or find the image " << filename << std::endl ;
          return false;
        }
        
        Eigen::Matrix4f tf = StringToMatrix4f(tfStr);
        //std::cout << tf << std::endl;

        KeyframeContainer* kfc;
        if(load_descs && desc_filename != "")
        {
          std::stringstream ss;
          ss << keyframes.size();
          
          Mat descriptors;
          std::vector<KeyPoint> keypoints;
          
          fs[std::string("descriptors") + ss.str()] >> descriptors;
          FileNode kpn = fs[std::string("keypoints") + ss.str()];
          read(kpn, keypoints); 
          kfc = new KeyframeContainer(img, tf, keypoints, descriptors);
        }
        else
        {  
          kfc = new KeyframeContainer(img, tf);
        }
        keyframes.push_back(kfc);    
      }
      //std::cout << "Found chunk " << chunk->Attribute("label") << std::endl;
    }
  }
  std::cout << "done" << std::endl;

  if(load_descs)
  {
    fs.release();
  }
  else if(desc_filename != "")
  {
    WriteDescriptorsToFile(desc_filename);
  }
  return true;
}

bool MapLocalizer::WriteDescriptorsToFile(std::string filename)
{
  FileStorage fs(filename, FileStorage::WRITE);
  
  std::cout << "Saving keypoints and descriptors..." << std::flush;
  if(!fs.isOpened())
  {
    std::cout << "Could not open descriptor file " << filename << std::endl;
    return false;
  }

  for(int i = 0; i < keyframes.size(); i++)
  {
    std::stringstream ss;
    ss << i;
    write(fs, std::string("keypoints") + ss.str(), keyframes[i]->GetKeypoints());
    write(fs, std::string("descriptors") + ss.str(), keyframes[i]->GetDescriptors());
  }
  fs.release();
  std::cout << "done" << std::endl;

  return true;
}

std::vector< KeyframeMatch > MapLocalizer::FindImageMatches(KeyframeContainer* img, int k)
{
  const double numMatchThresh = 0;//0.16;
  const double matchRatio = 0.8;
  std::vector< KeyframeMatch > kfMatches;
  
  // Find potential frame matches
  for(unsigned int i = 0; i < keyframes.size(); i++)
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

Eigen::Matrix4f MapLocalizer::FindImageTf(KeyframeContainer* img, std::vector< KeyframeMatch > matches, std::vector< KeyframeMatch >& goodMatches, std::vector< Eigen::Vector3f >& goodTVecs)
{
  Eigen::Matrix4f tf = Eigen::MatrixXf::Identity(4,4);
  std::vector<Eigen::Matrix4f> goodPs;
  goodMatches.clear();
  goodTVecs.clear();

  for(int i = 0; i < matches.size(); i++)
  {
    double km[3][3] = {{701.907522339299, 0, 352.73599016194}, {0, 704.43277859417, 230.636873050629}, {0, 0, 1}};
    //double dm[5] = {0,0,0,0,0};
    double dm[5] = {-0.456758192707853, 0.197636354824418, 0.000543685887014507, 0.000401738655456894, 0};
    Mat K(3,3, CV_64FC1, km);
    Mat dist = Mat(3,1, CV_64FC1, dm);
    Matx34d P(1,0,0,0,
              0,1,0,0, 
              0,0,1,0);
    Matx34d P1(1,0,0,0,
              0,1,0,0,
              0,0,1,0);

    std::vector<KeyPoint> matchPts1_good;
    std::vector<KeyPoint> matchPts2_good;
    std::vector<DMatch> ptmatches = matches[i].matches;
    std::vector<CloudPoint> outCloud;

    bool goodF = FindCameraMatrices(K, K.inv(), dist, img->GetKeypoints(), matches[i].kfc->GetKeypoints(), matchPts1_good, matchPts2_good, P, P1, ptmatches, outCloud);
    std::cout << "goodF: " << goodF << std::endl; 
    if(goodF)
    {
      Eigen::Matrix4f goodP;
      goodP << P1(0,0), P1(0,1), P1(0,2), P1(0,3),
               P1(1,0), P1(1,1), P1(1,2), P1(1,3),
               P1(2,0), P1(2,1), P1(2,2), P1(2,3),
                     0,       0,       0,       1;
      goodPs.push_back(goodP);
      goodMatches.push_back(matches[i]);
    } 
  }
  if(goodPs.size() < 2)
  {
    std::cout << "Not enough good matches.  Found " << goodPs.size() << std::endl;
    return tf;
  } 
  // http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/
  std::cout << "--------------------" << std::endl; 
  std::cout << "# of good matches: " << goodPs.size() << "/" << matches.size() << std::endl; 
  std::cout << "Actual Tf: " << std::endl << img->GetTf() << std::endl;
  Eigen::MatrixXf lsA(3*goodPs.size(), 3);
  Eigen::VectorXf lsb(3*goodPs.size());
  for(int i = 0; i < goodPs.size(); i++)
  {
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
    std::cout << "--------------------" << std::endl; 
    std::cout << imgWorldtf << std::endl << std::endl;
    //std::cout << goodMatches[i].kfc->GetTf().transpose()*goodPs[i] << std::endl << std::endl;
  }
  Eigen::VectorXf lsTrans = lsA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(lsb);
  
  std::cout << "--------------------" << std::endl; 
  std::cout << "Least squares T:" << std::endl << lsTrans << std::endl; 
  tf.block<3,1>(0,3) = lsTrans;
  return tf;
}

/*void MapLocalizer::RtFromE(const Eigen::MatrixXf& E, Eigen::Matrix3f& R1, Eigen::Matrix3f& R2, Eigen::Vector3f& t1, Eigen::Vector3f& t2)
{

  //SVD svd(E);
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);

  Eigen::Matrix3f W;
  W << 0,-1,0,   //HZ 9.13
       1,0,0,
       0,0,1;
  Eigen::Matrix3f Wt;
  Wt << 0,1,0,
       -1,0,0,
        0,0,1;
  R1 = svd.matrixU() * W * svd.matrixV().transpose(); //HZ 9.19
  R2 = svd.matrixU() * Wt * svd.matrixV().transpose(); //HZ 9.19
  t1 = svd.matrixU().col(2); 
  t2 = -svd.matrixU().col(2); 
}
*/
//void MapLocalizer::TriangulatePoints()
//{
//}
