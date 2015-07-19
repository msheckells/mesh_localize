#include "mesh_localize/MeshLocalizer.h"

int main (int argc, char **argv)
{
  ros::init (argc, argv, "mesh_localize");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  MeshLocalizer ml(nh, nh_private);
  ros::spin ();
  return 0;
}
