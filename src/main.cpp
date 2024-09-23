/*
	Last Change : 2024.09.03
*/

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <chrono>

#include "../include/scrfd_detector.h"
#include "../include/multiobject_tracker.h"
#include "../include/utils.h"
#include "../include/cluster.h"

bool clustering_done = false;
std::vector<std::vector<FaceObject>> initialFaceBoxes;
std::vector<cv::Rect> clusteredRects;
std::unordered_map<int, FaceTracker> faceTrackers;
std::unordered_map<int, cv::Rect> faceTrackerIDs;

int processFile(const std::string& filePath, SCRFD* detector, \
                const std::filesystem::path& resultDir) {
    if (isImageFile(filePath)) { // Process image
        cv::Mat img = cv::imread(filePath);
        std::cout << "[INFO] Processing image: " << filePath << std::endl;
        if (img.empty()) {
            fprintf(stderr, "[ERROR] cv::imread %s failed\n", filePath.c_str());
            return EXIT_FAILURE;
        }

        std::vector<FaceObject> faceobjects;
        float prob_threshold = 0.5f;  // 人脸置信度阈值
        float nms_threshold = 0.4f;   // 非极大值抑制阈值

        detector->detect(img, faceobjects, prob_threshold, nms_threshold);
        detector->draw(img, faceobjects);  // 绘制人脸框

        std::string output_path = (resultDir / (std::filesystem::path(filePath).filename().replace_extension(".png"))).string();
        cv::imwrite(output_path, img);

        std::cout << "[INFO] Image saved to: " << output_path << std::endl;
    } else if (isVideoFile(filePath)) { // Process video
        cv::VideoCapture cap(filePath);
        if (!cap.isOpened()) {
            return EXIT_FAILURE;
        }

        cv::Mat frame;
        int interval = 1;
        const int standard = 60; // 设定的标准帧数量

        int processed_frames = 0;
        const int update_interval = 10; // 更新进度条
        
        int frame_width =  static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::string output_path = (
            resultDir / (std::filesystem::path(filePath).filename().replace_extension(".mp4"))).string();
        // cv::VideoWriter video( \
        //     output_path, cv::VideoWriter::fourcc('X', '2', '6', '4'), cap.get(cv::CAP_PROP_FPS), cv::Size(frame_width, frame_height));
        cv::VideoWriter video( \
            output_path, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), cap.get(cv::CAP_PROP_FPS), cv::Size(frame_width, frame_height));
        
        int numClusters = 0;
        float prob_threshold = 0.3f;  // 人脸置信度阈值
        float nms_threshold = 0.3f;   // 非极大值抑制阈值

        MultiObjectTracker tracker;
        std::vector<FaceObject> faceobjects;
        std::map<int, std::vector<cv::Rect>> idToBoxes; // 用于存储每个ID对应的检测框
        std::map<int, cv::Rect> idToStandardBox;        // 用于存储每个ID的标准框

        while (cap.read(frame)) {
            if (frame.empty()) 
            {
                break;
            }
            std::vector<TrackingBox> detFrameData;

            if (processed_frames <= standard) // 如果在标准帧范围之内进行检测并存下检测信息用作后续聚类
            {
                detector->detect(frame, faceobjects, prob_threshold , nms_threshold);
                detector->draw(frame, faceobjects);
                
                std::vector<cv::Rect> detected_boxes;
                for (auto& face : faceobjects)
                {
                    detected_boxes.push_back(face.rect);
                }

                for (int i = 0 ; i < detected_boxes.size() ; i++)
                {
                    TrackingBox cur_box;
                    cur_box.box = detected_boxes[i];
                    cur_box.id = i;
                    cur_box.frame = processed_frames;
                    detFrameData.push_back(cur_box);
                }

                std::vector<TrackingBox> tracking_results = tracker.update(detFrameData);

                // std::cout << "tracking_results.size() : " << tracking_results.size() << std::endl;
                

                for (TrackingBox it : tracking_results) {
                    cv::rectangle(frame, it.box , (255,0,255), 2);
                    std::string label = "ID: " + std::to_string(it.id);
                    cv::putText(frame, label, cv::Point2f(it.box.x , it.box.y), 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255));
                    idToBoxes[it.id].emplace_back(it.box);
                }

                initialFaceBoxes.push_back(faceobjects);
                if (processed_frames == standard) 
                {
                    for (const auto& [id , boxes] : idToBoxes)
                    {
                        int numClusters = 1; // 根据ID的检测框进行聚类
                        cv::Rect standardBox = clusterFacesForID(boxes , numClusters);
                        idToStandardBox[id] = standardBox;
                    }
                    clustering_done = true;
                }
            } else {
                if ((processed_frames - standard) % interval == 0)
                {
                    detector->detect(frame, faceobjects, prob_threshold , nms_threshold);
                    detector->draw(frame, faceobjects);
                }
                if (clustering_done)
                {
                    for (const auto& [id , standardBox] : idToStandardBox) {
                        cv::rectangle(frame, standardBox, cv::Scalar(0, 255, 0), 2);
                        std::string label = "Standard ID: " + std::to_string(id);
                        cv::putText(frame, label, cv::Point2f(standardBox.x , standardBox.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                    }
                }
            }

            video.write(frame);

            // Update and print progress bar
            processed_frames++;
            if (processed_frames % update_interval == 0) {
                float progress = (float)processed_frames / total_frames;
                printProgressBar(progress);
            }
        }

        cap.release();
        video.release();

        std::cout << "[INFO] Video saved to: " << output_path << std::endl;
        std::cout << std::endl; // Move to the next line after the progress bar

        return EXIT_SUCCESS;
    }
} 

int main(int argc, char** argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [directory_path] \n", argv[0]);
        return EXIT_FAILURE;
    }

    const std::string dirPath = argv[1];

    std::string param_path = "{$YourParamPath}";
    param_path = "..\\assets\\scrfd_10g-opt2.param";
    std::string model_path = "{$YourModelPath}";
    model_path = "..\\assets\\scrfd_10g-opt2.bin";

    if (!std::filesystem::exists(dirPath) || !std::filesystem::is_directory(dirPath)) {
        std::cerr << "[ERROR] File does not exist: " << dirPath << std::endl;
        return EXIT_FAILURE;
    }
    if (!std::filesystem::exists(param_path) || !std::filesystem::exists(model_path)) {
        std::cerr << "[ERROR] Model File or Param File does not exist" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::filesystem::path resultDir = std::filesystem::path(dirPath) / "result";
    if (!std::filesystem::exists(resultDir)) {
        std::filesystem::create_directory(resultDir); 
    }

    SCRFD *detector = new SCRFD();
    detector->load(param_path, model_path, false);

    for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
        if (entry.is_regular_file()) {
            std::string filePath = entry.path().string();
            processFile(filePath, detector, resultDir);
        }
    }

    delete detector;

    return EXIT_SUCCESS;
}



