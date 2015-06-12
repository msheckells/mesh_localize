#include "map_localize/ImageDbUtil.h"

#include <iomanip>
#include <fstream>
#include <opencv2/core/eigen.hpp>

Eigen::Matrix4f ImageDbUtil::StringToMatrix4f(std::string str)
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

bool ImageDbUtil::LoadOgreDataDir(std::string data_dir, std::vector<KeyframeContainer*>& keyframes)
{
  printf("Opening keypoints and descriptors...");
  std::fstream fsk;
  fsk.open((data_dir + "/" + "keyframe000.jpg").c_str());
  int num_keyframes = 0;
  while(fsk.is_open()) 
  {
    num_keyframes++;
    fsk.close();
    std::stringstream ss;
    ss << data_dir << "/" << "keyframe" << std::setw(3) << std::setfill('0') << num_keyframes << ".jpg";
    fsk.open(ss.str().c_str());
  }

  if(num_keyframes == 0)
  {
    printf("No data in directory %s", data_dir.c_str());
    return false;
  }

  std::cout << "Successfully opened " << num_keyframes << " keypoints and descriptors" << std::endl;

  std::stringstream ss;
  Mat Kcv;
  Eigen::MatrixXf K;
  ss.str(std::string()); // cleaning ss
  ss << data_dir << "/intrinsics.xml";
  FileStorage fs_K(ss.str(), FileStorage::READ);
  fs_K["intrinsics"] >> Kcv;
  fs_K.release(); 
  cv2eigen(Kcv, K);

  for(int i = 0; i < num_keyframes; i++)
  {
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "keyframe" << std::setw(3) << std::setfill('0') << i << ".jpg";
    Mat keyframe = imread(ss.str());

    Mat pose;
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "pose" << std::setw(3) << std::setfill('0') << i << ".xml";
    FileStorage fs_pose(ss.str(), FileStorage::READ);
    fs_pose["pose"] >> pose;
    fs_pose.release(); 
 
    Mat desc;
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "descriptors" << std::setw(3) << std::setfill('0') << i << ".xml";
    FileStorage fs_desc(ss.str(), FileStorage::READ);
    fs_desc["descriptors"] >> desc;
    fs_desc.release(); 

    std::vector<KeyPoint> kps;
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "keypoints" << std::setw(3) << std::setfill('0') << i << ".xml";
    FileStorage fs_kp(ss.str(), FileStorage::READ);
    FileNode kn = fs_kp["keypoints"];
    read(kn, kps);
    fs_kp.release();

    Mat depth;
    ss.str(std::string()); // cleaning ss
    ss << data_dir << "/" << "depth" << std::setw(3) << std::setfill('0') << i << ".xml";
    FileStorage fs_depth(ss.str(), FileStorage::READ);
    fs_depth["depth"] >> depth;
    fs_depth.release(); 
 
    Eigen::MatrixXf pose_eig;
    cv2eigen(pose, pose_eig);
    //std::cout << "pose=" << std::endl << pose << std::endl;
    //std::cout << "pose_eig=" << std::endl << pose_eig << std::endl;
    //std::cout << "Kcv=" << std::endl << Kcv << std::endl;
    //std::cout << "K=" << std::endl << K << std::endl;
    CameraContainer* cc = new CameraContainer(keyframe, Eigen::Matrix4f(pose_eig), K);
    KeyframeContainer* kfc = new KeyframeContainer(cc, kps, desc, depth);
    keyframes.push_back(kfc);    
  }
  std::cout << "Successfully loaded model keyframes" << std::endl;
  return true;
}

bool ImageDbUtil::LoadPhotoscanFile(std::string filename, vector<CameraContainer*>& cameras, Mat map_Kcv, Mat map_distcoeffcv)
{
  TiXmlDocument doc(filename);

  printf("Loading %s...\n", filename.c_str());
  if(!doc.LoadFile())
  {  
    printf("Failed to load photoscan file\n");
    return false;
  }
  printf("Successfully loaded photoscan file\n");

  printf("Loading images...\n");
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
          printf("Could not open or find the image %s\n", filename.c_str());
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
        Eigen::MatrixXf map_K;
        cv2eigen(map_Kcv, map_K); 
        cameras.push_back(new CameraContainer(img_undistort, tf, map_K));
      }
      //std::cout << "Found chunk " << chunk->Attribute("label") << std::endl;
    }
  }
  return true;
}
