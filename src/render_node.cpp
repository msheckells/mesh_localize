#include <iostream>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "mesh_localize/OgreImageGenerator.h"

using namespace cv;

int main (int argc, char **argv)
{
  ros::init (argc, argv, "render");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  double step_size;

  if(!nh_private.getParam("step_size", step_size))
    step_size = 0.1;

  std::string resource_path = ros::package::getPath("mesh_localize");
  resource_path += "/ogre_cfg/";
  OgreImageGenerator oig(resource_path,"box.mesh");
  Eigen::Matrix3f K = oig.GetK();

  ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/virtual_camera/image",1000);
  ros::Publisher depth_pub = nh.advertise<sensor_msgs::Image>("/virtual_camera/depth",1000);
  ros::Publisher info_pub = nh.advertise<sensor_msgs::CameraInfo>("/virtual_camera/camera_info",1000);
  ros::Rate loop_rate(60);

  double x = 0;
  double y = 0;
  double z = 10;
  double xybound = 4;
  double zubound = 12;
  double zlbound = 8;
  while(ros::ok())
  {
    Mat depth, mask;
    Eigen::Matrix4f pose;

    x += (double(rand())/RAND_MAX)*step_size - step_size/2;    
    y += (double(rand())/RAND_MAX)*step_size - step_size/2;    
    z += (double(rand())/RAND_MAX)*step_size - step_size/2;    
    x = std::max(x, -xybound);
    y = std::max(y, -xybound);
    z = std::max(z, zlbound);
    x = std::min(x, xybound);
    y = std::min(y, xybound);
    z = std::max(z, zubound);
    pose << 1, 0, 0, x,
            0, -1, 0, y,
           0, 0, -1, z,
          0, 0, 0, 1;
    //pose <<   0.97152, -0.0578314,   0.229792,    0.29792,
    //     -3.78345e-10,    0.96976,   0.244059,    2.6059,
    //        -0.236957,  -0.237108,   0.942142,    -9.42142,
    //                0,          0,          0,          1;
    Mat img = oig.GenerateVirtualImage(pose, depth, mask);
    cv_bridge::CvImage cv_img;
    cv_img.image = img;
    cv_img.encoding = sensor_msgs::image_encodings::MONO8;

    cv_bridge::CvImage cv_depth;
    cv_depth.image = depth;
    cv_depth.encoding = sensor_msgs::image_encodings::TYPE_32FC1;

    sensor_msgs::CameraInfo ci_msg;
    ci_msg.header.stamp = ros::Time::now();
    ci_msg.height = img.rows;
    ci_msg.width = img.cols;
    ci_msg.distortion_model = "plumb_bob";
    for(int i = 0; i < 5; i++)
    {
      ci_msg.D.push_back(0); 
    }
    ci_msg.K[0] = K(0,0); ci_msg.K[1] = K(0,1); ci_msg.K[2] = K(0,2);
    ci_msg.K[3] = K(1,0); ci_msg.K[4] = K(1,1); ci_msg.K[5] = K(1,2);
    ci_msg.K[6] = K(2,0); ci_msg.K[7] = K(2,1); ci_msg.K[8] = K(2,2);

    info_pub.publish(ci_msg);
    image_pub.publish(cv_img.toImageMsg());
    depth_pub.publish(cv_depth.toImageMsg());
    img.release();
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
