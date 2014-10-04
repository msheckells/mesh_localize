#include "map_localize/GazeboLocalizer.h"

int main (int argc, char **argv)
{
  ros::init (argc, argv, "gazebo_localize");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  GazeboLocalizer gl(nh, nh_private);
  ros::spin ();
  return 0;
}
