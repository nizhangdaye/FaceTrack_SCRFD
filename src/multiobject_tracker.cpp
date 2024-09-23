/*
	Last Change : 2024.09.03
*/

#include "../include/multiobject_tracker.h"

MultiObjectTracker::MultiObjectTracker(int max_age , double iouThreshold)
{
    this->max_missing_frames = max_age;
    FaceTracker::kf_count = 0;
}

double MultiObjectTracker::calculateIoU(const cv::Rect_<float> rect1 , const cv::Rect_<float> rect2) 
{
    float in = (rect1 & rect2).area();
    float un = rect2.area() + rect1.area() - in;
    if (un <DBL_EPSILON)
    {
        return 0;
    }

    return (double)(in / un);
}

std::vector<TrackingBox> MultiObjectTracker::update(const std::vector<TrackingBox> &detFrameData) 
{
    total_frames++;
    frame_count++;
    int64 start_time = cv::getTickCount();

    if (trackers.size() == 0) // initialize kalman trackers using first detections.
    {
        for (unsigned int i = 0; i < detFrameData.size(); i++)
        {
            FaceTracker trk = FaceTracker(detFrameData[i].box);
            trackers.push_back(trk);
        }
        return std::vector<TrackingBox>();
    }
    
    // get predicted locations from existing trackers.
    predictedBoxes.clear();

    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Rect_<float> pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else // 对于已经产生非法的predict的跟踪器进行erase
        {
            it = trackers.erase(it);
        }
    }

    // associate detections to tracked object (both represented as bounding boxes)
    trkNum = predictedBoxes.size();
    detNum = detFrameData.size();

    iouMatrix.clear();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - calculateIoU(predictedBoxes[i], detFrameData[j].box);
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);

    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();

    if (detNum > trkNum) //	如果检测框数量大于预测框数量，计算未匹配的检测框。
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);
        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        std::set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(), \
            insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum) // 如果预测框数量大于检测框数量，找出未匹配的跟踪框（assignment[i] == -1）
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }

    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i); //将IOU低于阈值iouThreshold的匹配对标记为未匹配。
            unmatchedDetections.insert(assignment[i]);
        }
        else // 有效的匹配对（IOU 高于阈值）被存储在 matchedPairs 中
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    // updating trackers
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].update(detFrameData[detIdx].box);
    }

    // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        FaceTracker tracker = FaceTracker(detFrameData[umd].box);
        trackers.push_back(tracker);
    }

    // get trackers' output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).time_since_update < 2) && \
            ((*it).continual_hits >= min_hits || frame_count <= min_hits))
        {
            TrackingBox res;
            res.box = (*it).lastState;
            res.id = (*it).id + 1;
            // std::cout << "(*it).id : " << (*it).id << "-- res.id : " << res.id << std::endl;
            res.frame = frame_count;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).time_since_update > max_missing_frames)
            it = trackers.erase(it);
    }

    cycle_time = (double)(cv::getTickCount() - start_time);
    total_time += cycle_time / cv::getTickFrequency();

    return frameTrackingResult;
}


void MultiObjectTracker::removeInactiveTrackers()
{
    
}
