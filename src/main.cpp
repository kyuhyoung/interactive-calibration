#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <string>
#include <exception>
#include <iostream>

#include "calibCommon.hpp"
#include "calibPipeline.hpp"
#include "frameProcessor.hpp"

using namespace calib;

const char* keys  =
        "{n        | 20      | Number of frames for calibration }"
        "{v        |         | Input from video file }"
        "{ci       | 0       | DefaultCameraID }"
        "{si       | false   | Save captured frames }"
        "{flip       | false   | Vertical flip of input frames }"
        "{t        | circles | Template for calibration (circles, chessboard, dualCircles, chAruco) }";

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    captureParameters capParams;

    capParams.flipVertical = parser.get<bool>("flip");

    if (parser.has("v")) {
        capParams.source = InputVideoSource::File;
        capParams.videoFileName = parser.get<std::string>("v");
    }
    else {
        capParams.source = InputVideoSource::Camera;
        capParams.camID = parser.get<int>("ci");
    }

    auto templateType = parser.get<std::string>("t");

    if(templateType.find("circles", 0) == 0) {
        capParams.board = TemplateType::AcirclesGrid;
        capParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("chessboard", 0) == 0) {
        capParams.board = TemplateType::Chessboard;
        capParams.boardSize = cv::Size(7, 5);
    }
    else if(templateType.find("dualCircles", 0) == 0) {
        capParams.board = TemplateType::DoubleAcirclesGrid;
        capParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("chAruco", 0) == 0)
        capParams.board = TemplateType::chAruco;

    Sptr<calibrationData> globalData(new calibrationData);
    Sptr<FrameProcessor> capProcessor, showProcessor;
    capProcessor = Sptr<FrameProcessor>(new CalibProcessor(globalData, capParams.board, capParams.boardSize));
    showProcessor = Sptr<FrameProcessor>(new ShowProcessor(globalData));

    Sptr<CalibPipeline> pipeline(new CalibPipeline(capParams));
    std::vector<Sptr<FrameProcessor>> processors;
    processors.push_back(capProcessor);
    processors.push_back(showProcessor);


    try {
        while(true)
        {
            //collect data
            if (pipeline->start(processors) != 0)
                break;
            //start calibration
            globalData->imageSize = pipeline->getImageSize();
            std::cout << "calibration started\n";
            if(capParams.board != TemplateType::chAruco)
                globalData->totalAvgErr = cv::calibrateCamera(globalData->objectPoints, globalData->imagePoints, globalData->imageSize, globalData->cameraMatrix,
                                globalData->distCoeffs, globalData->rvecs, globalData->tvecs, 0, cv::TermCriteria(
                                    cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-5) );
            else {
                cv::Ptr<cv::aruco::Dictionary> dictionary =
                        cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(0));
                cv::Ptr<cv::aruco::CharucoBoard> charucoboard =
                            cv::aruco::CharucoBoard::create(6, 8, 200, 100, dictionary);
                globalData->totalAvgErr =
                        cv::aruco::calibrateCameraCharuco(globalData->allCharucoCorners, globalData->allCharucoIds, charucoboard, globalData->imageSize,
                                                      globalData->cameraMatrix, globalData->distCoeffs, globalData->rvecs, globalData->tvecs, 0, cv::TermCriteria(
                                                              cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 1e-5));
            }
            //show undist view
            //if(pipeline->start(showProcessor)!= 0)
            for (auto it = processors.begin(); it != processors.end(); ++it)
                        (*it)->resetState();

            //capProcessor->resetState();
            //showProcessor->resetState();
        }
        //write calibration data to file
    }
    catch (std::runtime_error exp)
    {
        std::cout << exp.what() << std::endl;
    }

    return 0;
}
