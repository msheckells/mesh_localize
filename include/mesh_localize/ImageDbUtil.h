#ifndef _IMAGE_DB_UTIL_H_
#define _IMAGE_DB_UTIL_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tinyxml.h>

#include "CameraContainer.h"
#include "KeyframeContainer.h"

class ImageDbUtil
{
public:
  
  static Eigen::Matrix4f StringToMatrix4f(std::string str);
  static bool LoadPhotoscanFile(std::string filename, std::vector<CameraContainer*>& cameras, Mat map_Kcv, Mat map_distcoeffcv);
  static bool LoadOgreDataDir(std::string data_dir, std::vector<KeyframeContainer*>& keyframes);
};
#endif
