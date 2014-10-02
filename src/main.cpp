/**
 * @file map_localize/src/main.cpp
 *
 * @brief Test Program
 **/
/*****************************************************************************
** Includes
*****************************************************************************/

#include <ros/ros.h>
#include <std_msgs/String.h>
#include "../include/map_localize/map_localize.hpp"

/*****************************************************************************
** Main
*****************************************************************************/

int main(int argc, char **argv) {

  map_localize::Foo foo;
  foo.helloDude();

  /*********************
  ** Example Talker
  **********************/
  ros::init(argc,argv,"test_map_localize");
  ros::NodeHandle nh;
  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);
  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
    std_msgs::String msg;
    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();
    ROS_INFO_STREAM("map_localize : " << msg.data);
    chatter_pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
    ++count;
    }

  return 0;
}
