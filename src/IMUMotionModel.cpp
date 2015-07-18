#include "map_localize/IMUMotionModel.h"

#include <tf/transform_broadcaster.h>

using namespace Eigen;
using namespace gcop;

IMUMotionModel::IMUMotionModel():
  estimated_bias(false),
  filtered_acc(0,0,0)
{
  // create custom callback queue for nodehandle
  nh.setCallbackQueue(&callback_queue);

  cam_transform <<  0,  -1, 0,
                    0,   0, -1,
                    1,   0, 0;

  imu_sub = nh.subscribe<vi_estimator_msgs::ImuSimple>("imu", 1000, &IMUMotionModel::handleImu, 
    this, ros::TransportHints().tcpNoDelay());
  //timer_send_tf = nh.createTimer(ros::Duration(0.05), &IMUMotionModel::publishTf, this);

  // create separate thread to get IMU data
  callback_thread = boost::thread(&IMUMotionModel::monitorQueue, this);
}

void IMUMotionModel::monitorQueue()
{
  ros::Time lastTime = ros::Time::now();
  while (ros::ok())
  {
    callback_queue.callAvailable(ros::WallDuration(0.1));
    if((ros::Time::now() - lastTime).toSec() > 0.01)
    {
      publishTf();
      lastTime = ros::Time::now();
    }
  }
}

void IMUMotionModel::reset()
{
  ins_state.P.topLeftCorner<3,3>().diagonal().setConstant(100);  // R
  ins_state.P.block<3,3>(3,3).diagonal() = cam_transform*gyro_bias_var;     // bg
  ins_state.P.block<3,3>(6,6).diagonal() = cam_transform*acc_bias_var;    // ba
  ins_state.P.block<3,3>(9,9).diagonal().setConstant(100); // p
  ins_state.P.block<3,3>(12,12).diagonal().setConstant(.01);    // v
  ins_state.v = Vector3d(0,0,0);
  ins_state.bg = cam_transform*gyro_bias;
  ins_state.ba = cam_transform*acc_bias;

  {
    boost::mutex::scoped_lock lock(data_store_mutex);
    w_store.clear();
    a_store.clear();
    dt_store.clear();
  }
}

void IMUMotionModel::init(const Eigen::Matrix4f& tf, const Matrix6f& cov)
{
  ins_state.P.topLeftCorner<3,3>() = cov.topLeftCorner<3,3>().cast<double>();
  ins_state.P.block<3,3>(3,3).diagonal() = cam_transform*gyro_bias_var;     // bg
  ins_state.P.block<3,3>(6,6).diagonal() = cam_transform*acc_bias_var;    // ba
  ins_state.P.block<3,3>(9,9) = cov.bottomRightCorner<3,3>().cast<double>();
  ins_state.P.block<3,3>(12,12).diagonal().setConstant(.01);    // v
  ins_state.R = tf.block<3,3>(0,0).cast<double>();
  ins_state.p = tf.block<3,1>(0,3).cast<double>();
  ins_state.v = Vector3d(0,0,0);
  ins_state.bg = cam_transform*gyro_bias;
  ins_state.ba = cam_transform*acc_bias;
 
  //std::cout << "init: " << tf.block<3,1>(0,3).transpose() << std::endl;

  {
    boost::mutex::scoped_lock lock(data_store_mutex);
    w_store.clear();
    a_store.clear();
    dt_store.clear();
  }
}

Eigen::Matrix4f IMUMotionModel::predict()
{
  
  std::vector<Vector6d> us;
  std::vector<double> dt_store_copy;
  {
    boost::mutex::scoped_lock lock(data_store_mutex);
    dt_store_copy = dt_store;
    us.resize(dt_store_copy.size());
    for(int i = 0; i < dt_store.size(); i++)
    {
      Vector6d u;
      u << w_store[i], a_store[i];
      us[i] = u;
    }
    
    w_store.clear();
    a_store.clear();
    dt_store.clear();
    
  }
  
  for(int i = 0; i < dt_store_copy.size(); i++)
  {
    kp_ins->Predict(ins_state, 0, ins_state, us[i], dt_store_copy[i]); 
    std::cout << "u["<<i<<"]="<<us[i].transpose() << std::endl;;
    std::cout << "position predict P["<<i<<"]="<<ins_state.P.block<3,3>(9,9).diagonal().transpose() << std::endl;;
    std::cout << "position predict p["<<i<<"]="<<ins_state.p.transpose() << std::endl;
  }
  
  Eigen::Matrix4f tf;
  tf.setIdentity();
  tf.block<3,3>(0,0) = ins_state.R.cast<float>();
  tf.block<3,1>(0,3) = ins_state.p.cast<float>();
  std::cout << "inverse predict pos=" << tf.inverse().block<3,1>(0,3).transpose() << std::endl;
  return tf;
}

Eigen::Matrix4f IMUMotionModel::correct(const Eigen::Matrix4f& tf, const Matrix6f& cov)
{
  PoseState pstate;
  pstate.R = tf.block<3,3>(0,0).cast<double>();
  pstate.p = tf.block<3,1>(0,3).cast<double>();
  kc_inspose->sensor.R = cov.cast<double>();
  kc_inspose->Correct(ins_state, 0, ins_state, Vector6d(), pstate);
  std::cout << "position correct P="<<ins_state.P.block<3,3>(9,9).diagonal().transpose() << std::endl;;
  
  // Reset state to corrected
  ins_state.R = tf.block<3,3>(0,0).cast<double>();
  ins_state.p = tf.block<3,1>(0,3).cast<double>();
  ins_state.P.topLeftCorner<3,3>() = cov.topLeftCorner<3,3>().cast<double>();  // R
  ins_state.P.block<3,3>(9,9) = cov.block<3,3>(3,3).cast<double>(); // p
  ins_pre_correct_state = ins_state;

  Eigen::Matrix4f ret_tf;
  ret_tf.setIdentity();
  ret_tf.block<3,3>(0,0) = ins_state.R.cast<float>();
  ret_tf.block<3,1>(0,3) = ins_state.p.cast<float>();
  std::cout << "inverse correct pos=" << ret_tf.inverse().block<3,1>(0,3).transpose() << std::endl;
  return ret_tf;
}

void IMUMotionModel::initSensors()
{
  gyro_bias.setZero();
  gyro_bias_var.setZero();
  acc_bias.setZero();
  acc_bias_var.setZero();
  for(int i = 0; i < bias_samples.size(); i++)
  {
    gyro_bias += bias_samples[i];
    acc_bias += acc_bias_samples[i];
  }
  gyro_bias /= bias_samples.size();
  acc_bias /= acc_bias_samples.size();

  for(int i = 0; i < bias_samples.size(); i++)
  {
    gyro_bias_var += (gyro_bias-bias_samples[i]).cwiseProduct(gyro_bias-bias_samples[i]);
    acc_bias_var += (acc_bias-acc_bias_samples[i]).cwiseProduct(acc_bias-acc_bias_samples[i]);
  }
  gyro_bias_var /= bias_samples.size();
  acc_bias_var /= acc_bias_samples.size();

  // Init sensors + EKFs
  double sv = sqrt((gyro_bias_var[0] + gyro_bias_var[1] + gyro_bias_var[2])/9.);
  double su = 1e-8;
  double sra = sqrt((acc_bias_var[0] + acc_bias_var[1] + acc_bias_var[2])/9.);
  ins.sv = sv;
  ins.su = su;
  ins.sra = 10*sra;
  ins.g0 << 0,0,0; // We will compensate for gravity ourselves using the other ImuEkf
  
  imu.sv = sv;
  imu.su = su;
  imu_sensor.a0[2] = acc_gravity;
  imu_sensor.sra = sra;//0.1;
  imu_sensor.R.topLeftCorner<3,3>().diagonal().setConstant(imu_sensor.sra*imu_sensor.sra);
  kp_ins = new InsKalmanPredictor(ins);
  kc_inspose = new InsPoseKalmanCorrector(ins.X, inspose);
  kp_imu = new ImuKalmanPredictor(imu);
  kc_imu = new ImuKalmanCorrector(imu.X, imu_sensor);

  ins_state.p = Vector3d(0,0,0);
  ins_state.v = Vector3d(0,0,0);
  ins_state.P.topLeftCorner<3,3>().diagonal().setConstant(100);  // R
  ins_state.P.block<3,3>(3,3).diagonal() = cam_transform*gyro_bias_var;     // bg
  ins_state.P.block<3,3>(6,6).diagonal() = cam_transform*acc_bias_var;    // ba
  ins_state.P.block<3,3>(9,9).diagonal().setConstant(100); // p
  ins_state.P.block<3,3>(12,12).diagonal().setConstant(100);    // v
  ins_state.bg = cam_transform*gyro_bias;
  ins_state.ba = cam_transform*acc_bias;

  imu_state.R.setIdentity();
  imu_state.P.topLeftCorner<3,3>().diagonal().setConstant(0.01);  // R
  imu_state.P.block<3,3>(3,3).diagonal() = gyro_bias_var;     // bg
  imu_state.P.block<3,3>(6,6).diagonal() = acc_bias_var;     // ba
  imu_state.bg = gyro_bias;
  imu_state.ba = acc_bias;
  std::cout << "gyro_bias: " << gyro_bias.transpose() << std::endl;
  std::cout << "gyro_bias_var: " << gyro_bias_var.transpose() << std::endl;
  std::cout << "acc_bias: " << acc_bias.transpose() << std::endl;
  std::cout << "acc_bias_var: " << acc_bias_var.transpose() << std::endl;

  t_epoch_0 = ros::Time::now();
  t_epoch_prev = t_epoch_0;
  estimated_bias = true;
}

void IMUMotionModel::storeMeasurements(Vector3d a, Vector3d w, double dt)
{
  {
    boost::mutex::scoped_lock lock(data_store_mutex);
    a_store.push_back(a);
    w_store.push_back(w);
    dt_store.push_back(dt);
  }
}

bool IMUMotionModel::isCalibrated()
{
  return estimated_bias;
}

void IMUMotionModel::handleImu(const vi_estimator_msgs::ImuSimpleConstPtr& msg_imu)
{

  double ax =  msg_imu->ax;
  double ay =  msg_imu->ay;
  double az =  msg_imu->az;
  double wx =  msg_imu->gx;
  double wy =  msg_imu->gy;
  double wz =  msg_imu->gz;

  Vector3d a(ax,ay,az);
  a *= acc_gravity;
  Vector3d w(wx,wy,wz);

  if(estimated_bias)
  {
    double t  = (msg_imu->t_epoch - t_epoch_0).toSec();
    double dt = (msg_imu->t_epoch - t_epoch_prev).toSec();

    //filtered_acc = 0.05*a + 0.95*filtered_acc;
    //a = filtered_acc;
    ImuState imu_state_predict;
    //kp_imu->Predict(imu_state, t, imu_state, w, dt);
    kp_imu->Predict(imu_state_predict, t, imu_state, w, dt);
    if(abs(a.norm() - acc_gravity) < 2*acc_gravity)
      kc_imu->Correct(imu_state, t, imu_state_predict, w, a);
    //std::cout << "a = " << a.transpose() << std::endl;
    //std::cout << "bg = " << imu_state.bg.transpose() << std::endl;

    Vector6d u;
    Vector3d w_cam = cam_transform*w;
    Vector3d a_cam_gcomp = cam_transform*imu_state.R.transpose()*(imu_state.R*a - Vector3d(0, 0, acc_gravity));
    u << w_cam, a_cam_gcomp;
    kp_ins->Predict(ins_pre_correct_state, t, ins_pre_correct_state, u, dt);
    storeMeasurements(a_cam_gcomp, w_cam, dt);
    //std::cout << "a_cam = " << a_cam_gcomp.transpose() << std::endl;
    //std::cout << "dt = " << dt << std::endl;
    t_epoch_prev = msg_imu->t_epoch;
  }
  else
  {
    bias_samples.push_back(w);
    acc_bias_samples.push_back(a - Vector3d(0,0,acc_gravity));
    // calc gyro bias and stddev
    if(bias_samples.size() >= num_bias_samples)
    {
      initSensors();
    }
  }

}

void IMUMotionModel::publishTf()
{
  tf::Transform trfm_rot, trfm_ins_no_trans, trfm_ins;
  trfm_rot.setOrigin( tf::Vector3(0,0,0) );
  trfm_ins_no_trans.setOrigin( tf::Vector3(0,0,0) );
  trfm_ins.setOrigin( tf::Vector3(ins_pre_correct_state.p(0),ins_pre_correct_state.p(1),
    ins_pre_correct_state.p(2)));

  Vector3d rpy;
  tf::Quaternion q;

  gcop::SO3::Instance().g2q(rpy, imu_state.R);
  q.setRPY(rpy[0],rpy[1],rpy[2]);
  trfm_rot.setRotation(q);

  gcop::SO3::Instance().g2q(rpy, ins_pre_correct_state.R);
  q.setRPY(rpy[0],rpy[1],rpy[2]);
  trfm_ins_no_trans.setRotation(q);
  trfm_ins.setRotation(q);

  static tf::TransformBroadcaster br;
  br.sendTransform(tf::StampedTransform(trfm_rot, ros::Time::now(), "world", "map_localizer/imu"));
  br.sendTransform(tf::StampedTransform(trfm_ins_no_trans, ros::Time::now(), "world", 
    "map_localizer/ins_no_trans"));
  br.sendTransform(tf::StampedTransform(trfm_ins, ros::Time::now(), "world", 
    "map_localizer/ins"));
  //std::cout << "v = " << ins_pre_correct_state.v.transpose() << std::endl;
}

