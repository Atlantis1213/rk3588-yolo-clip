#pragma once
#include "utils.hpp"

namespace yolo_clip {

inline std::vector<uint8_t> ReadBinaryFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open file: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(size);
  if (size > 0) {
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  }
  return bytes;
}

inline std::vector<std::string> LoadImagePaths(const std::string& dir) {
  DIR* handle = opendir(dir.c_str());
  if (!handle) {
    throw std::runtime_error("failed to open image dir: " + dir);
  }
  std::vector<std::string> paths;
  while (dirent* entry = readdir(handle)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    const auto dot = name.find_last_of('.');
    if (dot == std::string::npos) {
      continue;
    }
    std::string ext = name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
      paths.push_back(dir + "/" + name);
    }
  }
  closedir(handle);
  std::sort(paths.begin(), paths.end());
  return paths;
}

inline std::vector<cv::Mat> LoadImages(const std::string& dir) {
  const auto paths = LoadImagePaths(dir);
  std::vector<cv::Mat> images;
  for (const auto& path : paths) {
    cv::Mat image = cv::imread(path);
    if (!image.empty()) {
      images.push_back(image);
    }
  }
  if (images.empty()) {
    throw std::runtime_error("no images loaded from " + dir);
  }
  return images;
}

inline cv::VideoCapture OpenVideo(const std::string& path) {
  cv::VideoCapture cap;
  cap.open(path, cv::CAP_FFMPEG);
  if (cap.isOpened()) {
    return cap;
  }
  cap.open(path, cv::CAP_GSTREAMER);
  if (cap.isOpened()) {
    return cap;
  }
  cap.open("file://" + path);
  if (!cap.isOpened()) {
    throw std::runtime_error("failed to open video: " + path);
  }
  return cap;
}


}  // namespace yolo_clip
