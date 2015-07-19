#include "mesh_localize/MapFeatures.h"

MapFeatures::MapFeatures(std::vector<KeyframeContainer*>& kcv,  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) :
kcv(kcv), cloud(cloud)
{
  const int match_neighborhood = 1;
  const int maxDepth = 99999999;
  descriptors = Mat(0, kcv[0]->GetDescriptors().cols, CV_32F);

  for(unsigned int i = 0; i < kcv.size(); i++)
  {
    int height = kcv[i]->GetImage().size().height;
    int width = kcv[i]->GetImage().size().width;
    std::vector< std::vector< int > > idx_3dpt(width);
    std::vector< std::vector< double > > depths(width);
    for(int j = 0; j < width; j++)
    {
      idx_3dpt[j].resize(height, -1);
      depths[j].resize(height, maxDepth);
    }

    //Create depth map for keyframe
    Eigen::MatrixXf P(3,4);
    P = kcv[i]->GetK()*kcv[i]->GetTf().inverse().block<3,4>(0,0);
    for(unsigned int j = 0; j < cloud->points.size(); j++)
    {
      Eigen::Vector4f hpt(cloud->points[j].x, cloud->points[j].y, cloud->points[j].z, 1);
      Eigen::Vector3f impt = P*hpt;
      impt /= impt(2);
      if(impt(0) < 0  || impt(0) >= width || impt(1) < 0 || impt(1) >= height)
      {
        continue;
      }
      double depth = (kcv[i]->GetTf().inverse()*hpt)(2);
      int dx_idx = floor(impt(0));
      int dy_idx = floor(impt(1));
      if(depth > 0 && depth < depths[dx_idx][dy_idx])
      {
        depths[dx_idx][dy_idx] = depth;
        idx_3dpt[dx_idx][dy_idx] = j;
      }
    }
   
    // Find closest 3d pt to each image descriptor (only use if it's in some neighborhood)
    std::vector<KeyPoint> kps = kcv[i]->GetKeypoints();
    for(unsigned int j = 0; j < kps.size(); j++)
    {
      int closestPt = -1;
      int pt_x = kps[i].pt.x;
      int pt_y = kps[i].pt.y;
       
      for(int mn = 0; mn <= match_neighborhood && closestPt == -1; mn++)
      {
        for(int k = pt_x-mn; k <= pt_x+mn && closestPt == -1; k++)
        {
          for(int l = pt_y-mn; l <= pt_y+mn && closestPt == -1; l++)
          {
            if(k < 0 || k >= width || l < 0 || l >= height)
              continue;
            if(depths[k][l] < maxDepth)
            {
              closestPt = idx_3dpt[k][l];
            }
          }
        }
      }
      if(closestPt != -1)
      {
        keypoints.push_back(pcl::PointXYZ(cloud->points[closestPt].x, cloud->points[closestPt].y, cloud->points[closestPt].z));
        descriptors.push_back(kcv[i]->GetDescriptors().row(j));
      }
    }
  }
  std::cout << "Num Map Features: " << keypoints.size() << " " << descriptors.rows << std::endl;
}
 
Mat MapFeatures::GetDescriptors() const
{
  return descriptors;
}

std::vector<pcl::PointXYZ> MapFeatures::GetKeypoints() const
{
  return keypoints;
}

