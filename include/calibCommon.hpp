#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP

#include <memory>
#include <opencv2/core.hpp>
#include <vector>

namespace calib
{
    #define OVERLAY_DELAY 1000
    #define IMAGE_MAX_WIDTH 1280
    #define IMAGE_MAX_HEIGHT 960

    bool showOverlayMessage(const std::string& message);

    enum class InputType { Video, Pictures };
    enum class InputVideoSource { Camera, File };
    enum class TemplateType { AcirclesGrid, Chessboard, chAruco, DoubleAcirclesGrid };

    static const char* mainWindowName = "Calibration";
    static const char* gridWindowName = "Board locations";
    static const char* consoleHelp = "Hot keys:\nesc - exit application\n"
                              "s - save current data to .xml file\n"
                              "r - delete last frame\n"
                              "d - delete all frames";

    static const double sigmaMult = 1.96;

    struct calibrationData
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat stdDeviations;
        cv::Mat perViewErrors;
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        double totalAvgErr;
        cv::Size imageSize = cv::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);

        std::vector<std::vector<cv::Point2f>> imagePoints;
        std::vector< std::vector<cv::Point3f>> objectPoints;

        std::vector<cv::Mat> allCharucoCorners;
        std::vector<cv::Mat> allCharucoIds;

        cv::Mat undistMap1, undistMap2;
    };

    struct cameraParameters
    {
        cv::Mat cameraMatrix;
        cv::Mat distCoeffs;
        cv::Mat stdDeviations;
        double avgError;

        cameraParameters(){}
        cameraParameters(cv::Mat& _cameraMatrix, cv::Mat& _distCoeffs, cv::Mat& _stdDeviations, double _avgError = 0) :
            cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), stdDeviations(_stdDeviations), avgError(_avgError)
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
