#include "map_localize/MapLocalizer.h"
#include <tinyxml.h>
#include <algorithm>
#include <sstream>
#include <stdlib.h>     
#include <time.h> 

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
  Mat test = imread("/home/matt/uav_image_data/run9/frame0332.jpg", CV_LOAD_IMAGE_GRAYSCALE );
  KeyframeContainer* kf = new KeyframeContainer(test, Eigen::Matrix4f());
  //KeyframeContainer* kf = keyframes[rand() % keyframes.size()];
  std::vector< KeyframeMatch > matches = FindImageMatches(kf, 5);
  
  namedWindow( "Query", WINDOW_AUTOSIZE );// Create a window for display.
  imshow( "Query", kf->GetImage() ); 
  waitKey(0);
  for(int i = 0; i < matches.size(); i++)
  {
    namedWindow( "Match", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Match", matches[i].kfc->GetImage() ); 
    waitKey(0);
  }  
  /****/  
}

MapLocalizer::~MapLocalizer()
{
  for(int i = 0; i < keyframes.size(); i++)
  {
    delete keyframes[i];
  }
  keyframes.clear();
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
    std::cout << i/double(keyframes.size()) << std::endl;

    FlannBasedMatcher matcher;
    std::vector < std::vector< DMatch > > matches;
    matcher.knnMatch( img->GetDescriptors(), keyframes[i]->GetDescriptors(), matches, 2 );

    std::vector< DMatch > goodMatches;
    std::vector<Point2f> matchPts1;
    std::vector<Point2f> matchPts2;
    
    // Use ratio test to find good keypoint matches
    for(unsigned int j = 0; j < matches.size(); j++)
    {
      if(matches[j][0].distance < matchRatio*matches[j][1].distance)
      {
        goodMatches.push_back(matches[j][0]);
        matchPts1.push_back(img->GetKeypoints()[matches[j][0].queryIdx].pt);
        matchPts2.push_back(keyframes[i]->GetKeypoints()[matches[j][0].trainIdx].pt);
      }
    }
    if(goodMatches.size() >= numMatchThresh*matches.size())
    {
      std:: cout << "Found Match!" << std::endl;
      kfMatches.push_back(KeyframeMatch(keyframes[i], goodMatches, matchPts1, matchPts2));
    }
  }

  k = (kfMatches.size() < k) ? kfMatches.size() : k;
  std::sort(kfMatches.begin(), kfMatches.end());

  return std::vector< KeyframeMatch > (kfMatches.begin(), kfMatches.begin()+k);
}

Eigen::Matrix4f MapLocalizer::FindImageTf(KeyframeContainer* img, std::vector< KeyframeMatch >)
{
  Eigen::Matrix4f tf;
  
  //Mat fundamental_matrix = findFundamentalMat(pts1, pts2, FM_RANSAC, 3, 0.99);

  return tf;
}
