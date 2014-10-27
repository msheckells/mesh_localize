#include "map_localize/MapFeatures.h"

MapFeatures::MapFeatures(std::vector<KeyframeContainer*>& kcv,  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) :
kcv(kcv), cloud(cloud)
{
  
  for(unsigned int i = 0; i < kcv.size(); i++)
  {
    int height = kcv[i]->GetImage().size().height;
    int width = kcv[i]->GetImage().size().width;
    int** idx_3dpt = new int*[width];
    double** depths = new double*[width];
    for(int j = 0; j < width; j++)
    {
      idx_3dpt[j] = new int[height];
      depths[j] = new double[height];
      for(int k = 0; k < height; k++)
      {
        depths[j][k] = 999999999;
      }
    }

    //Create depth map for keyframe
    Eigen::MatrixXf P(3,4);
    P = kcv[i]->GetK()*kcv[i]->GetTf().inverse().block<3,4>(0,0);
    for(unsigned int j = 0; j < cloud->points.size(); j++)
    {
      Eigen::Vector4f hpt(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z, 1);
      Eigen::Vector3f impt = P*hpt;
      if(impt(0) < 0  || impt(0) >= width || impt(1) < 0 || impt(1) >= height)
      {
        continue;
      }
      double depth = (kcv[i]->GetTf().inverse()*hpt)(2);
      int dx_idx = floor(impt(0));
      int dy_idx = floor(impt(1));
      if(depth < depths[dx_idx][dy_idx])
      {
        depths[dx_idx][dy_idx] = depth;
        idx_3dpt[dx_idx][dy_idx] = j;
      }
    }
  }
}
 
Mat MapFeatures::GetDescriptors()
{
  return descriptors;
}

std::vector<pcl::PointXYZ> MapFeatures::GetKeypoints()
{
  return keypoints;
}

