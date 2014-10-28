#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

class ASiftDetector
{
public: 
  ASiftDetector();

  void detectAndCompute(const Mat& img, std::vector< KeyPoint >& keypoints, Mat& descriptors);

private:
  void affineSkew(double tilt, double phi, Mat& img, Mat& mask, Mat& Ai);  
};
