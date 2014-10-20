/*****************************************************************************
*   ExploringSfMWithOpenCV
******************************************************************************
*   by Roy Shilkrot, 5th Dec 2012
*   http://www.morethantechnical.com/
******************************************************************************
*   Ch4 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#include "map_localize/FindCameraMatrices.h"
#include "map_localize/Triangulation.h"

#include <vector>
#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

#ifdef USE_EIGEN
#include <Eigen/Eigen>
#endif

#define DECOMPOSE_SVD

#ifndef CV_PCA_DATA_AS_ROW
#define CV_PCA_DATA_AS_ROW 0
#endif

void DecomposeEssentialUsingHorn90(double _E[9], double _R1[9], double _R2[9], double _t1[3], double _t2[3]) {
	//from : http://people.csail.mit.edu/bkph/articles/Essential.pdf
#ifdef USE_EIGEN
	using namespace Eigen;

	Matrix3d E = Map<Matrix<double,3,3,RowMajor> >(_E);
	Matrix3d EEt = E * E.transpose();
	Vector3d e0e1 = E.col(0).cross(E.col(1)),e1e2 = E.col(1).cross(E.col(2)),e2e0 = E.col(2).cross(E.col(2));
	Vector3d b1,b2;

#if 1
	//Method 1
	Matrix3d bbt = 0.5 * EEt.trace() * Matrix3d::Identity() - EEt; //Horn90 (12)
	Vector3d bbt_diag = bbt.diagonal();
	if (bbt_diag(0) > bbt_diag(1) && bbt_diag(0) > bbt_diag(2)) {
		b1 = bbt.row(0) / sqrt(bbt_diag(0));
		b2 = -b1;
	} else if (bbt_diag(1) > bbt_diag(0) && bbt_diag(1) > bbt_diag(2)) {
		b1 = bbt.row(1) / sqrt(bbt_diag(1));
		b2 = -b1;
	} else {
		b1 = bbt.row(2) / sqrt(bbt_diag(2));
		b2 = -b1;
	}
#else
	//Method 2
	if (e0e1.norm() > e1e2.norm() && e0e1.norm() > e2e0.norm()) {
		b1 = e0e1.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else if (e1e2.norm() > e0e1.norm() && e1e2.norm() > e2e0.norm()) {
		b1 = e1e2.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	} else {
		b1 = e2e0.normalized() * sqrt(0.5 * EEt.trace()); //Horn90 (18)
		b2 = -b1;
	}
#endif
	
	//Horn90 (19)
	Matrix3d cofactors; cofactors.col(0) = e1e2; cofactors.col(1) = e2e0; cofactors.col(2) = e0e1;
	cofactors.transposeInPlace();
	
	//B = [b]_x , see Horn90 (6) and http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
	Matrix3d B1; B1 <<	0,-b1(2),b1(1),
						b1(2),0,-b1(0),
						-b1(1),b1(0),0;
	Matrix3d B2; B2 <<	0,-b2(2),b2(1),
						b2(2),0,-b2(0),
						-b2(1),b2(0),0;

	Map<Matrix<double,3,3,RowMajor> > R1(_R1),R2(_R2);

	//Horn90 (24)
	R1 = (cofactors.transpose() - B1*E) / b1.dot(b1);
	R2 = (cofactors.transpose() - B2*E) / b2.dot(b2);
	Map<Vector3d> t1(_t1),t2(_t2); 
	t1 = b1; t2 = b2;
	
	cout << "Horn90 provided " << endl << R1 << endl << "and" << endl << R2 << endl;
#endif
}

bool CheckCoherentRotation(cv::Mat_<double>& R) {

	
	if(fabsf(determinant(R))-1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix" << endl;
		return false;
	}
	return true;
}

Mat GetHomographyMat(const vector<KeyPoint>& imgpts1,
                                           const vector<KeyPoint>& imgpts2,
                                           vector<KeyPoint>& imgpts1_good,
                                           vector<KeyPoint>& imgpts2_good,
                                           vector<DMatch>& matches,
					   vector<DMatch>& nonmatches)
{
	vector<uchar> status(imgpts1.size());
	nonmatches.clear();
	
	std::vector< DMatch > good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	
	imgpts1_good.clear(); imgpts2_good.clear();
	
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0) { 
		//points already aligned...
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
	} else {
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}
	
	Mat H;
	{
		vector<Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
		
		double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		H = findHomography(pts1, pts2, CV_RANSAC, 0.006 * maxVal, status); //threshold from [Snavely07 4.1]
	}
	
	vector<DMatch> new_matches;
	cout << "H keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			if (matches.size() <= 0) { //points already aligned...
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			} else {
				new_matches.push_back(matches[i]);
			}
		}
		else
		{
			nonmatches.push_back(matches[i]);
		}
	}	
	
	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Homography Matrix\n";
	matches = new_matches; //keep only those points who survived the homography matrix
	
	return H;
}

Mat GetFundamentalMat(const vector<KeyPoint>& imgpts1,
					   const vector<KeyPoint>& imgpts2,
					   vector<KeyPoint>& imgpts1_good,
					   vector<KeyPoint>& imgpts2_good,
					   vector<DMatch>& matches
#ifdef __SFM__DEBUG__
					  ,const Mat& img_1,
					  const Mat& img_2
#endif
					  ) 
{
	//Try to eliminate keypoints based on the fundamental matrix
	//(although this is not the proper way to do this)
	vector<uchar> status(imgpts1.size());
	
#ifdef __SFM__DEBUG__
	std::vector< DMatch > good_matches_;
	std::vector<KeyPoint> keypoints_1, keypoints_2;
#endif		
	//	undistortPoints(imgpts1, imgpts1, cam_matrix, distortion_coeff);
	//	undistortPoints(imgpts2, imgpts2, cam_matrix, distortion_coeff);
	//
	imgpts1_good.clear(); imgpts2_good.clear();
	
	vector<KeyPoint> imgpts1_tmp;
	vector<KeyPoint> imgpts2_tmp;
	if (matches.size() <= 0) { 
		//points already aligned...
		imgpts1_tmp = imgpts1;
		imgpts2_tmp = imgpts2;
	} else {
		GetAlignedPointsFromMatch(imgpts1, imgpts2, matches, imgpts1_tmp, imgpts2_tmp);
	}
	
	Mat F;
	{
		vector<Point2f> pts1,pts2;
		KeyPointsToPoints(imgpts1_tmp, pts1);
		KeyPointsToPoints(imgpts2_tmp, pts2);
#ifdef __SFM__DEBUG__
		cout << "pts1 " << pts1.size() << " (orig pts " << imgpts1_tmp.size() << ")" << endl;
		cout << "pts2 " << pts2.size() << " (orig pts " << imgpts2_tmp.size() << ")" << endl;
#endif
		double minVal,maxVal;
		cv::minMaxIdx(pts1,&minVal,&maxVal);
		F = findFundamentalMat(pts1, pts2, FM_RANSAC, 0.006 * maxVal, 0.99, status); //threshold from [Snavely07 4.1]
	}
	
	vector<DMatch> new_matches;
	cout << "F keeping " << countNonZero(status) << " / " << status.size() << endl;	
	for (unsigned int i=0; i<status.size(); i++) {
		if (status[i]) 
		{
			imgpts1_good.push_back(imgpts1_tmp[i]);
			imgpts2_good.push_back(imgpts2_tmp[i]);

			if (matches.size() <= 0) { //points already aligned...
				new_matches.push_back(DMatch(matches[i].queryIdx,matches[i].trainIdx,matches[i].distance));
			} else {
				new_matches.push_back(matches[i]);
			}

#ifdef __SFM__DEBUG__
			good_matches_.push_back(DMatch(imgpts1_good.size()-1,imgpts1_good.size()-1,1.0));
			keypoints_1.push_back(imgpts1_tmp[i]);
			keypoints_2.push_back(imgpts2_tmp[i]);
#endif
		}
	}	
	
	cout << matches.size() << " matches before, " << new_matches.size() << " new matches after Fundamental Matrix\n";
	matches = new_matches; //keep only those points who survived the fundamental matrix
	
#if 0
	//-- Draw only "good" matches
#ifdef __SFM__DEBUG__
	if(!img_1.empty() && !img_2.empty()) {		
		vector<Point2f> i_pts,j_pts;
		Mat img_orig_matches;
		{ //draw original features in red
			vector<uchar> vstatus(imgpts1_tmp.size(),1);
			vector<float> verror(imgpts1_tmp.size(),1.0);
			img_1.copyTo(img_orig_matches);
			KeyPointsToPoints(imgpts1_tmp, i_pts);
			KeyPointsToPoints(imgpts2_tmp, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,0,255));
		}
		{ //superimpose filtered features in green
			vector<uchar> vstatus(imgpts1_good.size(),1);
			vector<float> verror(imgpts1_good.size(),1.0);
			i_pts.resize(imgpts1_good.size());
			j_pts.resize(imgpts2_good.size());
			KeyPointsToPoints(imgpts1_good, i_pts);
			KeyPointsToPoints(imgpts2_good, j_pts);
			drawArrows(img_orig_matches, i_pts, j_pts, vstatus, verror, Scalar(0,255,0));
			imshow( "Filtered Matches", img_orig_matches );
		}
		int c = waitKey(0);
		if (c=='s') {
			imwrite("fundamental_mat_matches.png", img_orig_matches);
		}
		destroyWindow("Filtered Matches");
	}
#endif		
#endif
	
	return F;
}

void TakeSVDOfE(Mat_<double>& E, Mat& svd_u, Mat& svd_vt, Mat& svd_w) {
#if 1
	//Using OpenCV's SVD
	SVD svd(E,SVD::MODIFY_A);
	svd_u = svd.u;
	svd_vt = svd.vt;
	svd_w = svd.w;
#else
	//Using Eigen's SVD
	cout << "Eigen3 SVD..\n";
	Eigen::Matrix3f  e = Eigen::Map<Eigen::Matrix<double,3,3,Eigen::RowMajor> >((double*)E.data).cast<float>();
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(e, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXf Esvd_u = svd.matrixU();
	Eigen::MatrixXf Esvd_v = svd.matrixV();
	svd_u = (Mat_<double>(3,3) << Esvd_u(0,0), Esvd_u(0,1), Esvd_u(0,2),
						  Esvd_u(1,0), Esvd_u(1,1), Esvd_u(1,2), 
						  Esvd_u(2,0), Esvd_u(2,1), Esvd_u(2,2)); 
	Mat_<double> svd_v = (Mat_<double>(3,3) << Esvd_v(0,0), Esvd_v(0,1), Esvd_v(0,2),
						  Esvd_v(1,0), Esvd_v(1,1), Esvd_v(1,2), 
						  Esvd_v(2,0), Esvd_v(2,1), Esvd_v(2,2));
	svd_vt = svd_v.t();
	svd_w = (Mat_<double>(1,3) << svd.singularValues()[0] , svd.singularValues()[1] , svd.singularValues()[2]);
#endif
	
	cout << "----------------------- SVD ------------------------\n";
	cout << "U:\n"<<svd_u<<"\nW:\n"<<svd_w<<"\nVt:\n"<<svd_vt<<endl;
	cout << "----------------------------------------------------\n";
}

bool TestTriangulation(const vector<CloudPoint>& pcloud, const Matx34d& P, vector<uchar>& status) {
	vector<Point3d> pcloud_pt3d = CloudPointsToPoints(pcloud);
	vector<Point3d> pcloud_pt3d_projected(pcloud_pt3d.size());
	
	Matx44d P4x4 = Matx44d::eye(); 
	for(int i=0;i<12;i++) P4x4.val[i] = P.val[i];
	
	perspectiveTransform(pcloud_pt3d, pcloud_pt3d_projected, P4x4);
	
	status.resize(pcloud.size(),0);
	for (int i=0; i<pcloud.size(); i++) {
		status[i] = (pcloud_pt3d_projected[i].z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);

	double percentage = ((double)count / (double)pcloud.size());
	cout << count << "/" << pcloud.size() << " = " << percentage*100.0 << "% are in front of camera" << endl;
	if(percentage < 0.75)
		return false; //less than 75% of the points are in front of the camera

	return true;
}

bool TestCoplanarity(const vector<CloudPoint>& pcloud, vector<int>& planeIdx, vector<int>& nonplaneIdx)
{
	nonplaneIdx.clear();
 	cv::Mat_<double> cldm(pcloud.size(),3);
	for(unsigned int i=0;i<pcloud.size();i++) {
		cldm.row(i)(0) = pcloud[i].pt.x;
		cldm.row(i)(1) = pcloud[i].pt.y;
		cldm.row(i)(2) = pcloud[i].pt.z;
	}

	cv::Mat_<double> mean;
	cv::PCA pca(cldm,mean,CV_PCA_DATA_AS_ROW);
	int num_inliers = 0;

	cv::Vec3d nrm = pca.eigenvectors.row(2); nrm = nrm / norm(nrm);
	cv::Vec3d x0 = pca.mean;
	double p_to_plane_thresh = sqrt(pca.eigenvalues.at<double>(2));

	for (int i=0; i<pcloud.size(); i++) {
		Vec3d w = Vec3d(pcloud[i].pt) - x0;
		double D = fabs(nrm.dot(w));

		if(D < p_to_plane_thresh){
			num_inliers++;
			planeIdx.push_back(i);
		}
                else nonplaneIdx.push_back(i);
	}

	cout << num_inliers << "/" << pcloud.size() << " are coplanar" << endl;
	return (double)num_inliers / (double)(pcloud.size()) > 0.85;
}

bool DecomposeHtoRandT(
	Mat_<double>& H,
	vector<KeyPoint>& imgpts1_good,
	vector<KeyPoint>& imgpts2_good,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2) 
{
	Mat svd_u, svd_vt, svd_w;
	Mat eig_E, eig_V;
	TakeSVDOfE(H,svd_u,svd_vt,svd_w);

	H = H/svd_w.at<double>(1); // normalize H with second singular val
        std::cout << "normalized H: " << std::endl << Mat(H) << std::endl;	

	Mat_<double> HtH = H.t()*H;
	
	int numPositive = 0;
	for(unsigned int i = 0; i < imgpts1_good.size(); i++)
	{
		Matx31d pt1(imgpts1_good[i].pt.x, imgpts1_good[i].pt.y, 1);
		Matx31d pt2(imgpts2_good[i].pt.x, imgpts2_good[i].pt.y, 1);
		//Make sure x_2^T*H*x_1 > 0 for most x
		if(Mat(Mat(pt2.t())*H*Mat(pt1)).at<double>(0) > 0)
			numPositive++;
	}
	if(numPositive < 0.5*imgpts1_good.size())
		H = -H;
	
        eigen(HtH,eig_E,eig_V);
	std::cout << "Sig: " << std::endl << Mat(eig_E) << std::endl;
	std::cout << "V: " << std::endl << Mat(eig_V) << std::endl;
	double sig1 = eig_E.at<double>(0,0);
	double sig2 = eig_E.at<double>(0,1);
	double sig3 = eig_E.at<double>(0,2);
	if(abs(sig1 - sig3) < 1e-6)
		return false;

	Mat u1 = (sqrt(1 - sig3*sig3)*eig_V.col(0) + sqrt(sig1*sig1 -1)*eig_V.col(2))/sqrt(sig1*sig1-sig3*sig3); 
	Mat u2 = (sqrt(1 - sig3*sig3)*eig_V.col(0) - sqrt(sig1*sig1 -1)*eig_V.col(2))/sqrt(sig1*sig1-sig3*sig3); 
	
	std::cout << "u1: " << std::endl << Mat(u1) << std::endl;
	std::cout << "u2: " << std::endl << Mat(u2) << std::endl;
	Mat U1 = Mat::zeros(3,3,CV_32FC1);
	Mat U2 = Mat::zeros(3,3,CV_32FC1);
	Mat W1 = Mat::zeros(3,3,CV_32FC1);
	Mat W2 = Mat::zeros(3,3,CV_32FC1);

	eig_V.col(1).copyTo(U1.col(0));
	u1.copyTo(U1.col(1));
	Mat(U1.col(0)).cross(U1.col(1)).copyTo(U1.col(2));
	
	eig_V.col(1).copyTo(U2.col(0));
	u2.copyTo(U2.col(1));
	Mat(U1.col(0)).cross(U1.col(1)).copyTo(U2.col(2));
	
	Mat(H*eig_V.col(1)).copyTo(W1.col(0));
	Mat(H*u1).copyTo(W1.col(1));
	Mat(H*eig_V.col(1)).cross(H*u1).copyTo(W1.col(2));

	Mat(H*eig_V.col(1)).copyTo(W2.col(0));
	Mat(H*u2).copyTo(W2.col(1));
	Mat(H*eig_V.col(1)).cross(H*u2).copyTo(W2.col(2));
	
	std::cout << "U1: " << std::endl << Mat(U1) << std::endl;
	std::cout << "U2: " << std::endl << Mat(U2) << std::endl;
	std::cout << "W1: " << std::endl << Mat(W1) << std::endl;
	std::cout << "W2: " << std::endl << Mat(W2) << std::endl;
	
	R1 = Mat(W1*U1.t());
	Mat N1 = Mat(eig_V.col(1)).cross(u1); //v2 x u1
	t1 = (H-R1)*N1;
	if(N1.at<double>(2,0) < 0) 
	{
		t1 = -t1;
	}
	
	R2 = Mat(W2*U2.t());
	Mat N2 = eig_V.col(1).cross(u2); //v2 x u2
	t2 = (H-R2)*N2;
	if(N2.at<double>(2,0) < 0) 
	{
		t2 = -t2;
	}
	return true;
}

bool DecomposeEtoRandT(
	Mat_<double>& E,
	Mat_<double>& R1,
	Mat_<double>& R2,
	Mat_<double>& t1,
	Mat_<double>& t2) 
{
#ifdef DECOMPOSE_SVD
	//Using HZ E decomposition
	Mat svd_u, svd_vt, svd_w;
	TakeSVDOfE(E,svd_u,svd_vt,svd_w);

	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(svd_w.at<double>(0) / svd_w.at<double>(1));
	if(singular_values_ratio>1.0) singular_values_ratio = 1.0/singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.6/*0.7*/) {
		cout << "singular values are too far apart\n";
		return false;
	}

	Matx33d W(0,-1,0,	//HZ 9.13
		1,0,0,
		0,0,1);
	Matx33d Wt(0,1,0,
		-1,0,0,
		0,0,1);
	R1 = svd_u * Mat(W) * svd_vt; //HZ 9.19
	R2 = svd_u * Mat(Wt) * svd_vt; //HZ 9.19
	t1 = svd_u.col(2); //u3
	t2 = -svd_u.col(2); //u3
#else
	//Using Horn E decomposition
	DecomposeEssentialUsingHorn90(E[0],R1[0],R2[0],t1[0],t2[0]);
#endif
	return true;
}

bool FindCameraMatricesWithH(const Mat& K, 
						const Mat& Kinv, 
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1,
						vector<DMatch>& matches
						) 
{
	//Find camera matrices
	{
		cout << "Find camera H matrices...";
	
		vector<DMatch> nonmatches, nonmatches2;	
		vector<DMatch> matches2;
		vector<KeyPoint> imgpts1_good2;
		vector<KeyPoint> imgpts2_good2;

		Mat H = GetHomographyMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches,nonmatches);
		if(matches.size() < 15/*30*//*100*/) { // || ((double)imgpts1_good.size() / (double)    imgpts1.size()) < 0.25
                	cerr << "not enough inliers after H matrix" << endl;
                	return false;
                }
		if(nonmatches.size() < 10/*20*/){
			cerr << "not enough points left for second plane" << endl;
			return false;
		}
		matches2 = nonmatches;
		Mat H2 = GetHomographyMat(imgpts1,imgpts2,imgpts1_good2,imgpts2_good2,matches2,nonmatches2);
		if(matches2.size() < 10/*20*/)
		{
			cerr << "could not find second plane: not enough inliers" << endl;
			return false;
		}	

		Mat_<double> G = Kinv * H * K;
		Mat_<double> G2 = Kinv * H2 * K;
		Mat_<double> R1(3,3), R1_2(3,3);
                Mat_<double> R2(3,3), R2_2(3,3);
                Mat_<double> t1(1,3), t1_2(1,3);
                Mat_<double> t2(1,3), t2_2(1,3);

		if(!DecomposeHtoRandT(G, imgpts1_good, imgpts2_good, R1, R2, t1, t2))
			return false;
		if(!DecomposeHtoRandT(G2, imgpts1_good2, imgpts2_good2, R1_2, R2_2, t1_2, t2_2))
			return false;

		P1 = Matx34d(R1(0,0),   R1(0,1),        R1(0,2),        t1(0),
                             R1(1,0),   R1(1,1),        R1(1,2),        t1(1),
                             R1(2,0),   R1(2,1),        R1(2,2),        t1(2));
		Matx34d P2 = Matx34d(R2(0,0),   R2(0,1),        R2(0,2),        t2(0),
                             R2(1,0),   R2(1,1),        R2(1,2),        t2(1),
                             R2(2,0),   R2(2,1),        R2(2,2),        t2(2));
		Matx34d P1_2 = Matx34d(R1_2(0,0),   R1_2(0,1),        R1_2(0,2),        t1_2(0),
                              R1_2(1,0),   R1_2(1,1),        R1_2(1,2),        t1_2(1),
                              R1_2(2,0),   R1_2(2,1),        R1_2(2,2),        t1_2(2));
		Matx34d P2_2 = Matx34d(R2_2(0,0),   R2_2(0,1),        R2_2(0,2),        t2_2(0),
                              R2_2(1,0),   R2_2(1,1),        R2_2(1,2),        t2_2(1),
                              R2_2(2,0),   R2_2(2,1),        R2_2(2,2),        t2_2(2));
		std::cout << "Ps:" << std::endl << Mat(P1) << std::endl << Mat(P2) << std::endl;
		std::cout << "Ps2:" << std::endl << Mat(P1_2) << std::endl << Mat(P2_2) << std::endl;
		vector<Mat_<double> > R1s, R2s, t1s, t2s;
		t1s.push_back(t1);
		t1s.push_back(t2);
		R1s.push_back(R1);
		R1s.push_back(R2);
		t2s.push_back(t1_2);
		t2s.push_back(t2_2);
		R2s.push_back(R1_2);
		R2s.push_back(R2_2);
		double minTheta = 99999999;
                Mat_<double> minR1(3,3), minR2(3,3), mint1(1,3), mint2(1,3);

		for(unsigned int i = 0; i < t1s.size(); i++)
		{
			for(unsigned int i = 0; i < t1s.size(); i++)
			{
				double theta = acos(t1s[i].dot(t2s[i])/(norm(t1s[i])*norm(t2s[i])));
				if(theta < minTheta)
				{
					minTheta = theta;
					mint1 = t1s[i];
					mint2 = t2s[i];
					minR1 = R1s[i];	
					minR2 = R2s[i];	
				}
			}
		}

		std::cout << "Min angle: " << minTheta*180.0/M_PI << std::endl;
		if(minTheta*180./M_PI > 40)
		{
			cerr << "Two homogrpahy solution is not compatible" << std::endl;
			return false;
		}
		Mat_<double> t_est = (mint1 + mint2)/2;
		//t_est.copyTo(P1.col(3));
		P1 = Matx34d(minR1(0,0),   minR1(0,1),        minR1(0,2),        t_est(0),
                             minR1(1,0),   minR1(1,1),        minR1(1,2),        t_est(1),
                             minR1(2,0),   minR1(2,1),        minR1(2,2),        t_est(2));

		std::cout << "P_est:" << std::endl << Mat(P1) << std::endl;
		return true;
	}
}

bool FindCameraMatrices(const Mat& K, 
						const Mat& Kinv, 
						const Mat& distcoeff,
						const vector<KeyPoint>& imgpts1,
						const vector<KeyPoint>& imgpts2,
						vector<KeyPoint>& imgpts1_good,
						vector<KeyPoint>& imgpts2_good,
						Matx34d& P,
						Matx34d& P1,
						vector<DMatch>& matches,
						vector<CloudPoint>& outCloud
#ifdef __SFM__DEBUG__
						,const Mat& img_1,
						const Mat& img_2
#endif
						) 
{
	//Find camera matrices
	{
		cout << "Find camera matrices...";
		double t = getTickCount();
		
		Mat F = GetFundamentalMat(imgpts1,imgpts2,imgpts1_good,imgpts2_good,matches
#ifdef __SFM__DEBUG__
								  ,img_1,img_2
#endif
								  );
		
		//if(matches.size() < 20 /*100*/) { // || ((double)imgpts1_good.size() / (double)imgpts1.size()) < 0.25
		//	cerr << "not enough inliers after F matrix" << endl;
		//	return false;
		//}
		
		//Essential matrix: compute then extract cameras [R|t]
		Mat_<double> E = K.t() * F * K; //according to HZ (9.12)

		//according to http://en.wikipedia.org/wiki/Essential_matrix#Properties_of_the_essential_matrix
		if(fabsf(determinant(E)) > 1e-07) {
			cout << "det(E) != 0 : " << determinant(E) << "\n";
			P1 = 0;
			return false;
		}
		
		Mat_<double> R1(3,3);
		Mat_<double> R2(3,3);
		Mat_<double> t1(1,3);
		Mat_<double> t2(1,3);

		//decompose E to P' , HZ (9.19)
		{			
			if (!DecomposeEtoRandT(E,R1,R2,t1,t2)) return false;

			if(determinant(R1)+1.0 < 1e-09) {
				//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
				cout << "det(R) == -1 ["<<determinant(R1)<<"]: flip E's sign" << endl;
				E = -E;
				DecomposeEtoRandT(E,R1,R2,t1,t2);
			}
			if (!CheckCoherentRotation(R1)) {
				cout << "resulting rotation is not coherent\n";
				P1 = 0;
				return false;
			}
			
			P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t1(0),
						 R1(1,0),	R1(1,1),	R1(1,2),	t1(1),
						 R1(2,0),	R1(2,1),	R1(2,2),	t1(2));
			cout << "Testing P1 " << endl << Mat(P1) << endl;
			
			vector<CloudPoint> pcloud,pcloud1; vector<KeyPoint> corresp;
			double reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
			double reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
			vector<uchar> tmp_status;
			//check if pointa are triangulated --in front-- of cameras for all 4 ambiguations
			if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
				P1 = Matx34d(R1(0,0),	R1(0,1),	R1(0,2),	t2(0),
							 R1(1,0),	R1(1,1),	R1(1,2),	t2(1),
							 R1(2,0),	R1(2,1),	R1(2,2),	t2(2));
				cout << "Testing P1 "<< endl << Mat(P1) << endl;

				pcloud.clear(); pcloud1.clear(); corresp.clear();
				reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
				reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
				
				if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
					if (!CheckCoherentRotation(R2)) {
						cout << "resulting rotation is not coherent\n";
						P1 = 0;
						return false;
					}
					
					P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t1(0),
								 R2(1,0),	R2(1,1),	R2(1,2),	t1(1),
								 R2(2,0),	R2(2,1),	R2(2,2),	t1(2));
					cout << "Testing P1 "<< endl << Mat(P1) << endl;

					pcloud.clear(); pcloud1.clear(); corresp.clear();
					reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
					reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
					
					if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
						P1 = Matx34d(R2(0,0),	R2(0,1),	R2(0,2),	t2(0),
									 R2(1,0),	R2(1,1),	R2(1,2),	t2(1),
									 R2(2,0),	R2(2,1),	R2(2,2),	t2(2));
						cout << "Testing P1 "<< endl << Mat(P1) << endl;

						pcloud.clear(); pcloud1.clear(); corresp.clear();
						reproj_error1 = TriangulatePoints(imgpts1_good, imgpts2_good, K, Kinv, distcoeff, P, P1, pcloud, corresp);
						reproj_error2 = TriangulatePoints(imgpts2_good, imgpts1_good, K, Kinv, distcoeff, P1, P, pcloud1, corresp);
						
						if (!TestTriangulation(pcloud,P1,tmp_status) || !TestTriangulation(pcloud1,P,tmp_status) || reproj_error1 > 100.0 || reproj_error2 > 100.0) {
							cout << "Shit." << endl; 
							return false;
						}
					}				
				}			
			}
			for (unsigned int i=0; i<pcloud.size(); i++) {
				outCloud.push_back(pcloud[i]);
			}
		}		
		
		t = ((double)getTickCount() - t)/getTickFrequency();
		cout << "Done. (" << t <<"s)"<< endl;
	}
	return true;
}
