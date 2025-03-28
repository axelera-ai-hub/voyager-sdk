// Copyright Axelera AI, 2023
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "TrackerFactory.h"

using namespace axtracker;

//************** Pre/post-process for OpenCV YOLO ***********
struct DetectedResult {
  int class_id;
  float score;
  cv::Rect bbox;
};

cv::Mat
preprocess(const cv::Mat &frame, const int &input_size = 640)
{
  cv::Mat blob;
  cv::dnn::blobFromImage(frame, blob, 1 / 255.0,
      cv::Size(input_size, input_size), cv::Scalar(), true, false);
  return blob;
}

std::vector<DetectedResult>
postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs, cv::dnn::Net &net,
    int backend, float confThreshold = 0.3, float nmsThreshold = 0.4)
{
  static std::vector<int> outLayers = net.getUnconnectedOutLayers();
  static std::string outLayerType = net.getLayer(outLayers[0])->type;

  std::vector<DetectedResult> results;
  if (outLayerType == "DetectionOutput") {
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    CV_Assert(outs.size() > 0);
    for (size_t k = 0; k < outs.size(); k++) {
      float *data = (float *) outs[k].data;
      for (size_t i = 0; i < outs[k].total(); i += 7) {
        float confidence = data[i + 2];
        if (confidence > confThreshold) {
          int left = (int) data[i + 3];
          int top = (int) data[i + 4];
          int right = (int) data[i + 5];
          int bottom = (int) data[i + 6];
          int width = right - left + 1;
          int height = bottom - top + 1;
          if (width <= 2 || height <= 2) {
            left = (int) (data[i + 3] * frame.cols);
            top = (int) (data[i + 4] * frame.rows);
            right = (int) (data[i + 5] * frame.cols);
            bottom = (int) (data[i + 6] * frame.rows);
            width = right - left + 1;
            height = bottom - top + 1;
          }
          // Skip 0th background class id.
          results.push_back({ (int) (data[i + 1]) - 1, confidence,
              cv::Rect(left, top, width, height) });
        }
      }
    }
  } else if (outLayerType == "Region") {
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float *data = (float *) outs[i].data;
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        if (confidence > confThreshold) {
          int centerX = (int) (data[0] * frame.cols);
          int centerY = (int) (data[1] * frame.rows);
          int width = (int) (data[2] * frame.cols);
          int height = (int) (data[3] * frame.rows);
          int left = centerX - width / 2;
          int top = centerY - height / 2;

          results.push_back({ classIdPoint.x, (float) confidence,
              cv::Rect(left, top, width, height) });
        }
      }
    }
  } else
    CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another
  // backends we need NMS in sample or NMS is required if number of outputs > 1
  if (outLayers.size() > 1
      || (outLayerType == "Region" && backend != cv::dnn::DNN_BACKEND_OPENCV)) {
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < results.size(); i++) {
      if (results[i].score >= confThreshold) {
        class2indices[results[i].class_id].push_back(i);
      }
    }
    std::vector<DetectedResult> nmsResults;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin();
         it != class2indices.end(); ++it) {
      std::vector<cv::Rect> localBoxes;
      std::vector<float> localConfidences;
      std::vector<size_t> classIndices = it->second;
      for (size_t i = 0; i < classIndices.size(); i++) {
        localBoxes.push_back(results[classIndices[i]].bbox);
        localConfidences.push_back(results[classIndices[i]].score);
      }
      std::vector<int> nmsIndices;
      cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
      for (size_t i = 0; i < nmsIndices.size(); i++) {
        size_t idx = nmsIndices[i];
        nmsResults.push_back({ it->first, localConfidences[idx], localBoxes[idx] });
      }
    }
    results = nmsResults;
  }

  return results;
}

//************** tools ***********

std::vector<ax::ObservedObject>
convertDetections(const std::vector<DetectedResult> &detections, const cv::Size &frameSize)
{
  std::vector<ax::ObservedObject> convertedDetections;
  for (const auto &det : detections) {
    ax::ObservedObject obs = ax::ObservedObject::FromLTWH(det.bbox.x,
        det.bbox.y, det.bbox.width, det.bbox.height, det.class_id, det.score,
        frameSize.width, frameSize.height);
    convertedDetections.push_back(obs);
  }
  return convertedDetections;
}

cv::Scalar
getColorForTracker(int trackId)
{
  // Use trackId to seed the random number generator for consistency
  srand(trackId);
  return cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
}

bool
fileExists(const std::string &filename)
{
  std::ifstream file(filename.c_str());
  return file.good();
}
//************** main ***********

int
main(int argc, char **argv)
{
  std::cout << "OpenCV version: " << CV_VERSION << std::endl;
  std::string videoPath = "/home/atang/repo/application.framework/media/traffic2_480p.mp4";

  std::string trackerTypeStr = "SORT";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--algo") {
      if (i + 1 < argc) { // Make sure we aren't at the end of argv!
        trackerTypeStr = argv[++i]; // Increment 'i' so we don't get the
                                    // argument as the next argv[i].
        std::transform(trackerTypeStr.begin(), trackerTypeStr.end(),
            trackerTypeStr.begin(), ::tolower);
      }
    } else {
      videoPath = arg;
    }
  }
  std::cout << "Tracker type: " << trackerTypeStr << std::endl;

  if (videoPath == "cam") {
    videoPath = "0";
  }
  cv::VideoCapture capture(videoPath);
  cv::Mat frame;

  if (!capture.isOpened()) {
    std::cerr << "Error opening video capture" << std::endl;
    return -1;
  }

  // may need version 4.8 which support ONNX Split, Slice, Clip (Relu6) and Conv
  // with auto_pad. cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov5n.onnx");
  std::string weightsFile = "yolov3-320.weights";
  if (!fileExists(weightsFile)) {
    std::string command = "wget -O " + weightsFile + " https://pjreddie.com/media/files/yolov3.weights";
    std::cout << "Downloading weights file...\n";
    int ret = system(command.c_str());
    if (ret != 0) {
      std::cerr << "Error downloading weights file" << std::endl;
      return -1;
    }
  }
  cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", weightsFile);
  int input_size = 320; // 416, 640

  TrackerParams params = {}; // default params
  if (trackerTypeStr == "scalarmot") {
    params = CreateTrackerParams({ { "maxLostFrames", 30 } });
  } else if (trackerTypeStr == "sort") {
    params = CreateTrackerParams({ { "det_thresh", 0.0f }, { "maxAge", 30 },
        { "minHits", 3 }, { "iouThreshold", 0.3f } });
  } else if (trackerTypeStr == "bytetrack" || trackerTypeStr == "oc-sort"
             || trackerTypeStr == "oc_sort") {
    // use default params
  }
  std::unique_ptr<ax::MultiObjTracker> tracker
      = CreateMultiObjTracker(trackerTypeStr, params);

  int backend = cv::dnn::DNN_BACKEND_DEFAULT;
  net.setPreferableBackend(backend);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(2);
  double fps = 0.0;
  while (capture.read(frame)) {
    if (frame.empty())
      break;

    cv::Mat blob = preprocess(frame, input_size);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    std::vector<DetectedResult> detections = postprocess(frame, outputs, net, backend);

    double start = cv::getTickCount();
    // Update the tracker with the converted detections
    std::vector<ax::ObservedObject> convertedDetections
        = convertDetections(detections, frame.size());

    const auto &activeTrackers = tracker->Update(convertedDetections);
    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
    if (duration > 0) {
      fps = 1.0 / duration;
    }

    std::cout << "Processing time of tracker: " << duration * 1000 << " ms; FPS: " << fps
              << "\tNumber of trackedObject: " << activeTrackers.size() << std::endl;

    for (const auto &tkr : activeTrackers) {
      // Draw the last bounding box of each track
      const auto &bbox = tkr.GetXyxy(frame.cols, frame.rows);
      int id = tkr.track_id;
      cv::Scalar color = getColorForTracker(id);

      cv::Rect pixelBbox(cv::Point(std::get<0>(bbox), std::get<1>(bbox)),
          cv::Point(std::get<2>(bbox), std::get<3>(bbox)));
      cv::rectangle(frame, pixelBbox, color, 2);
      cv::putText(frame, std::to_string(id), cv::Point(pixelBbox.x, pixelBbox.y - 5),
          cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

      // TODO: build history and draw trajectories
      // for (size_t i = 1; i < history.size(); i++)
      // {
      //   cv::Point prevCenter(
      //       (history[i - 1].x1 + history[i - 1].x2) * frame.cols / 2,
      //       (history[i - 1].y1 + history[i - 1].y2) * frame.rows / 2);
      //   cv::Point currCenter((history[i].x1 + history[i].x2) * frame.cols /
      //   2,
      //                        (history[i].y1 + history[i].y2) * frame.rows /
      //                        2);
      //   cv::line(frame, prevCenter, currCenter, color, 2);
      // }
    }

    cv::imshow("Tracking", frame);
    if (cv::waitKey(1) == 'q')
      break;
  }

  capture.release();
  cv::destroyAllWindows();
  return 0;
}
