#ifndef _IMU_MOTION_MODEL_H_
#define _IMU_MOTION_MODEL_H_

#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

#include "vi_estimator_msgs/ImuSimple.h"

#include <Eigen/Dense>

#include <gcop/kalmanpredictor.h>
#include <gcop/kalmancorrector.h>
#include <gcop/insmanifold.h>
#include <gcop/insimu.h>
#include <gcop/imu.h>
#include <gcop/imusensor.h>
#include <gcop/inspose.h>

using namespace gcop;

class IMUMotionModel
{
  typedef Eigen::Matrix<float, 6, 6> Matrix6f;
  typedef KalmanPredictor<InsState, 15, 6, Dynamic> InsKalmanPredictor;
  typedef KalmanCorrector<InsState, 15, 6, Dynamic, PoseState, 6> InsPoseKalmanCorrector;
  typedef KalmanPredictor<ImuState, 9, 3, Dynamic> ImuKalmanPredictor;
  typedef KalmanCorrector<ImuState, 9, 3, Dynamic, Matrix<double, 3, 1>, 3> ImuKalmanCorrector;

public:
  IMUMotionModel();
  Eigen::Matrix4f predict();
  Eigen::Matrix4f correct(const Eigen::Matrix4f& tf, const Matrix6f& cov);
  void reset();
  void init(const Eigen::Matrix4f& tf, const Matrix6f& cov);
  bool isCalibrated();

private:
  void handleImu(const vi_estimator_msgs::ImuSimpleConstPtr& msg_imu);
  void initSensors();
  void storeMeasurements(Vector3d a, Vector3d w, double dt);
  void publishTf();
  void monitorQueue();

  ros::NodeHandle nh;

  ros::Subscriber imu_sub;
  //ros::Timer timer_send_tf;

  InsKalmanPredictor* kp_ins;
  InsPoseKalmanCorrector* kc_inspose;
  ImuKalmanPredictor* kp_imu;
  ImuKalmanCorrector* kc_imu;

  Imu imu;                  // Imu for prediction
  ImuSensor<3> imu_sensor;  // Imu for correction
  Ins ins;                  // Prediction for pose tracker
  InsPose<> inspose;        // Correction step coming from pose tracker
  InsState ins_state; // Tracks full state with respect to tracked object
  InsState ins_pre_correct_state; // Tracks full state with respect to tracked object, prediction applied before correction
  ImuState imu_state; // Tracks rotation with respect to world frame 

  Vector3d filtered_acc;
  Vector3d gyro_bias, gyro_bias_var;
  Vector3d acc_bias, acc_bias_var;

  const double acc_gravity = 9.80665;
  const int num_bias_samples = 1500;
  bool estimated_bias;
  std::vector<Eigen::Vector3d> bias_samples, acc_bias_samples;
  Eigen::Matrix3d cam_transform;

  ros::Time t_epoch_0;
  ros::Time t_epoch_prev;

  std::vector<Eigen::Vector3d> a_store, w_store;
  std::vector<double> dt_store;

  boost::thread callback_thread;
  boost::mutex data_store_mutex;
  ros::CallbackQueue callback_queue;
};

#endif
