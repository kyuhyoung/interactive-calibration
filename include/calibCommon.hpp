#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace calib
{
    enum class InputType { Video, Pictures };
    enum class InputVideoSource { Camera, File };
    enum class TemplateType { AcirclesGrid, Chessboard, chAruco, DoubleAcirclesGrid };

    static const char* mainWindowName = "Calibration";

    struct calibrationData
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat stdDeviations;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        std::vector<float> reprojErrs;
        double totalAvgErr;
        cv::Size imageSize;

        std::vector<std::vector<cv::Point2f>> imagePoints;
        std::vector< std::vector<cv::Point3f>> objectPoints;

        std::vector<cv::Mat> allCharucoCorners;
        std::vector<cv::Mat> allCharucoIds;
    };

    struct cameraParameters
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;

        cameraParameters(){}
        cameraParameters(cv::Mat& _cameraMatrix, cv::Mat& _distCoeffs) :
            cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs)
        {}
    };

    struct captureParameters
    {
        InputType captureMethod;
        InputVideoSource source;
        TemplateType board;
        cv::Size boardSize;
        float captureDelay = 500.f;
        std::string videoFileName;
        bool flipVertical;
        int camID;
    };

template <typename T>
    using Sptr = std::shared_ptr<T>;
}

#endif
