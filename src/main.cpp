#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include <exception>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "calibCommon.hpp"
#include "calibPipeline.hpp"
#include "frameProcessor.hpp"
#include "cvCalibrationFork.hpp"
#include "calibController.hpp"
#include "parametersController.hpp"
#include "rotationConverters.hpp"

using namespace calib;

const char* keys  =
        "{n        | 20      | Number of frames for calibration }"
        "{v        |         | Input from video file }"
        "{ci       | 0       | DefaultCameraID }"
        "{si       | false   | Save captured frames }"
        "{flip     | false   | Vertical flip of input frames }"
        "{t        | circles | Template for calibration (circles, chessboard, dualCircles, chAruco) }"
        "{sz       | 163     | Distance between two nearest centers of circles or squares on calibration board}"
        "{dst      | 295     | Distance between white and black parts of daulCircles template}"
        "{w        |         | Width of template (in corners or circles)}"
        "{h        |         | Height of template (in corners or circles)}"
        "{of       | CamParams.xml | Output file name}"
        "{ft       | true    | Auto tuning of calibration flags}"
        "{vis      | grid    | Captured boards visualisation (grid, window)}"
        "{d        | 900     | Min delay between captures}"
        "{pf       | params.xml| Advanced application parameters}"
        "{help     |         | Print help}";

bool calib::showOverlayMessage(const std::string& message)
{
#ifdef HAVE_QT
    cv::displayOverlay(mainWindowName, message, OVERLAY_DELAY);
    return true;
#else
    return false;
#endif
}

void deleteButton(int state, void* data)
{
    (static_cast<Sptr<calibDataController>*>(data))->get()->deleteLastFrame();
    calib::showOverlayMessage("Last frame deleted");
}

void deleteAllButton(int state, void* data)
{
    (static_cast<Sptr<calibDataController>*>(data))->get()->deleteAllData();
    calib::showOverlayMessage("All frames deleted");
}

void undistortButton(int state, void* data)
{
    ShowProcessor* processor = static_cast<ShowProcessor*>(((Sptr<FrameProcessor>*)data)->get());
    processor->setUndistort(static_cast<bool>(state));
    calib::showOverlayMessage(std::string("Undistort is ") +
                       (static_cast<bool>(state) ? std::string("on") : std::string("off")));
}

void saveCurrentParamsButton(int state, void* data)
{
    if((static_cast<Sptr<calibDataController>*>(data))->get()->saveCurrentCameraParameters())
        calib::showOverlayMessage("Calibration parameters saved");
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")) {
        parser.printMessage();
        return 0;
    }
    std::cout << consoleHelp << std::endl;
    parametersController paramsController;

    if(!paramsController.loadFromParser(parser))
        return 0;

    captureParameters capParams = paramsController.getCaptureParameters();
    internalParameters intParams = paramsController.getInternalParameters();

    cv::TermCriteria solverTermCrit = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,
                                                       intParams.solverMaxIters, intParams.solverEps);
    Sptr<calibrationData> globalData(new calibrationData);

    int calibrationFlags = 0;
    Sptr<calibController> controller(new calibController(globalData, calibrationFlags,
                                                         parser.get<bool>("ft"), capParams.minFramesNum));
    Sptr<calibDataController> dataController(new calibDataController(globalData, capParams.maxFramesNum));
    dataController->setParametersFileName(parser.get<std::string>("of"));

    Sptr<FrameProcessor> capProcessor, showProcessor;
    capProcessor = Sptr<FrameProcessor>(new CalibProcessor(globalData, capParams));
    showProcessor = Sptr<FrameProcessor>(new ShowProcessor(globalData, controller, capParams.board));

    if(parser.get<std::string>("vis").find("window") == 0) {
        static_cast<ShowProcessor*>(showProcessor.get())->setVisualisationMode(visualisationMode::Window);
        cv::namedWindow(gridWindowName);
        cv::moveWindow(gridWindowName, 1280, 500);
    }

    Sptr<CalibPipeline> pipeline(new CalibPipeline(capParams));
    std::vector<Sptr<FrameProcessor>> processors;
    processors.push_back(capProcessor);
    processors.push_back(showProcessor);

    cv::namedWindow(mainWindowName);
    cv::moveWindow(mainWindowName, 10, 10);
#ifdef HAVE_QT
    cv::createButton("Delete last frame", deleteButton, &dataController, CV_PUSH_BUTTON);
    cv::createButton("Delete all frames", deleteAllButton, &dataController, CV_PUSH_BUTTON);
    cv::createButton("Undistort", undistortButton, &showProcessor, CV_CHECKBOX, false);
    cv::createButton("Save current parameters", saveCurrentParamsButton, &dataController, CV_PUSH_BUTTON);
#endif
    try {
        while(true)
        {
            auto exitStatus = pipeline->start(processors);
            if (exitStatus == PipelineExitStatus::Finished) {
                if(controller->getCommonCalibrationState())
                    saveCurrentParamsButton(0, &dataController);
                break;
            }
            else if (exitStatus == PipelineExitStatus::Calibrate) {

                dataController->rememberCurrentParameters();
                globalData->imageSize = pipeline->getImageSize();
                calibrationFlags = controller->getNewFlags();

                //using namespace std::chrono;
                //auto startPoint = high_resolution_clock::now();
                if(capParams.board != TemplateType::chAruco) {
                    globalData->totalAvgErr =
                            cvfork::calibrateCamera(globalData->objectPoints, globalData->imagePoints,
                                                    globalData->imageSize, globalData->cameraMatrix,
                                                    globalData->distCoeffs, cv::noArray(), cv::noArray(),
                                                    globalData->stdDeviations, globalData->perViewErrors,
                                                    calibrationFlags, solverTermCrit);
                }
                else {
                    cv::Ptr<cv::aruco::Dictionary> dictionary =
                            cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(capParams.charucoDictName));
                    cv::Ptr<cv::aruco::CharucoBoard> charucoboard =
                                cv::aruco::CharucoBoard::create(capParams.boardSize.width, capParams.boardSize.height,
                                                                capParams.charucoSquareLenght, capParams.charucoMarkerSize, dictionary);
                    globalData->totalAvgErr =
                            cvfork::calibrateCameraCharuco(globalData->allCharucoCorners, globalData->allCharucoIds,
                                                           charucoboard, globalData->imageSize,
                                                           globalData->cameraMatrix, globalData->distCoeffs,
                                                           cv::noArray(), cv::noArray(), globalData->stdDeviations,
                                                           globalData->perViewErrors, calibrationFlags, solverTermCrit);
                }
                //auto endPoint = high_resolution_clock::now();
                //std::cout << "Calibration time: " << (duration_cast<duration<double>>(endPoint - startPoint)).count() << "\n";

                dataController->updateUndistortMap();
                dataController->printParametersToConsole(std::cout);
                controller->updateState();
                dataController->filterFrames();
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == PipelineExitStatus::DeleteLastFrame) {
                deleteButton(0, &dataController);
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == PipelineExitStatus::DeleteAllFrames) {
                deleteAllButton(0, &dataController);
                static_cast<ShowProcessor*>(showProcessor.get())->updateBoardsView();
            }
            else if (exitStatus == PipelineExitStatus::SaveCurrentData) {
                saveCurrentParamsButton(0, &dataController);
            }
            else if (exitStatus == PipelineExitStatus::SwitchUndistort)
                static_cast<ShowProcessor*>(showProcessor.get())->switchUndistort();

            for (auto it = processors.begin(); it != processors.end(); ++it)
                (*it)->resetState();
        }
    }
    catch (std::runtime_error exp) {
        std::cout << exp.what() << std::endl;
    }

    return 0;
}
