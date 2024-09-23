/*
	Last Change : 2024.09.03
*/

#include <opencv2/core/base.hpp>

#include "../include/face_tracker.h"

int FaceTracker::kf_count = 0;

FaceTracker::FaceTracker(): \
            age(0) , time_since_update(0) , num_hits(0) , continual_hits(0)
{
    initialize(cv::Rect_<float>());
    id = kf_count;
    std::cout << "id = kf_count : " << id << std::endl;

}

FaceTracker::FaceTracker(cv::Rect_<float> initRect) : \
            age(0) , time_since_update(0) , num_hits(0) , continual_hits(0)
{
    id = kf_count++;
    initialize(initRect);
}

FaceTracker::~FaceTracker()
{
    state_history.clear();
}

void FaceTracker::initialize(const cv::Rect_<float> stateMat) 
{
    int stateNum = 7;
    int measureNum = 4;
    // Initialize Kalman Filter parameters
    kf.init(stateNum, measureNum , 0); // State dimension, measurement dimension, control dimension
    kf.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) << 1, 0, 0, 0, 1, 0, 0, \
						   0, 1, 0, 0, 0, 1, 0,
						   0, 0, 1, 0, 0, 0, 1,
						   0, 0, 0, 1, 0, 0, 0,
						   0, 0, 0, 0, 1, 0, 0,
						   0, 0, 0, 0, 0, 1, 0,
						   0, 0, 0, 0, 0, 0, 1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    measurement = cv::Mat::zeros(measureNum, 1, CV_32F);

    // initialize state vector with bounding box in [cx,cy,s,r] style
	kf.statePost.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
	kf.statePost.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
	kf.statePost.at<float>(2, 0) = stateMat.area();
	kf.statePost.at<float>(3, 0) = stateMat.width / stateMat.height;
}

cv::Rect_<float> FaceTracker::xysr2rect(float center_x , float center_y , float s , float r)
{
    float w = sqrt(s * r);
	float h = s / w;
	float x = (center_x - w / 2);
	float y = (center_y - h / 2);

	if (x < 0 && center_x > 0)
		x = 0;
	if (y < 0 && center_y > 0)
		y = 0;

	return cv::Rect_<float>(x, y, w, h);
}

cv::Rect_<float> FaceTracker::predict() 
{   
    cv::Mat prediction = kf.predict();
    age += 1;

    if (time_since_update > 0)
    {
        continual_hits = 0;
    }
    time_since_update += 1;

    cv::Rect_<float> predict_box = xysr2rect( \
        prediction.at<float>(0, 0), prediction.at<float>(1, 0), prediction.at<float>(2, 0), prediction.at<float>(3, 0));

    state_history.push_back(predict_box);

    return state_history.back();
}

void FaceTracker::update(const cv::Rect_<float> new_bbox) 
{
    this->lastState = new_bbox;

    time_since_update = 0;
    state_history.clear();
    num_hits += 1;
    continual_hits += 1;

    // measurement
	measurement.at<float>(0, 0) = new_bbox.x + new_bbox.width / 2;
	measurement.at<float>(1, 0) = new_bbox.y + new_bbox.height / 2;
	measurement.at<float>(2, 0) = new_bbox.area();
	measurement.at<float>(3, 0) = new_bbox.width / new_bbox.height;

    kf.correct(measurement);
}

cv::Rect_<float> FaceTracker::get_state()
{
    cv::Mat state = kf.statePost;
    return xysr2rect(state.at<float>(0,0) , state.at<float>(1,0) , state.at<float>(2,0) , state.at<float>(3,0));
}
