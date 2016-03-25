#include "parametersController.hpp"
#include <iostream>

bool calib::parametersController::loadFromFile(const std::string &inputFileName)
{
    cv::FileStorage reader;
    reader.open(inputFileName, cv::FileStorage::READ);
    if(!reader.isOpened()) {
        std::cerr << "Unable to open " << inputFileName << std::endl;
        return false;
    }






    reader.release();
    return true;
}

calib::parametersController::parametersController()
{
}

calib::captureParameters calib::parametersController::getCaptureParameters() const
{
    return mCapParams;
}

bool calib::parametersController::loadFromParser(cv::CommandLineParser &parser)
{
    mCapParams.flipVertical = parser.get<bool>("flip");
    mCapParams.captureDelay = parser.get<float>("d");

    if (parser.has("v")) {
        mCapParams.source = InputVideoSource::File;
        mCapParams.videoFileName = parser.get<std::string>("v");
    }
    else {
        mCapParams.source = InputVideoSource::Camera;
        mCapParams.camID = parser.get<int>("ci");
    }

    auto templateType = parser.get<std::string>("t");

    if(templateType.find("circles", 0) == 0) {
        mCapParams.board = TemplateType::AcirclesGrid;
        mCapParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("chessboard", 0) == 0) {
        mCapParams.board = TemplateType::Chessboard;
        mCapParams.boardSize = cv::Size(7, 7);
    }
    else if(templateType.find("dualcircles", 0) == 0) {
        mCapParams.board = TemplateType::DoubleAcirclesGrid;
        mCapParams.boardSize = cv::Size(4, 11);
    }
    else if(templateType.find("charuco", 0) == 0) {
        mCapParams.board = TemplateType::chAruco;
        mCapParams.boardSize = cv::Size(6, 8);
    }

    if(parser.has("w") && parser.has("h")) {
        mCapParams.boardSize = cv::Size(parser.get<int>("w"), parser.get<int>("h"));
        if (mCapParams.boardSize.width <= 0 || mCapParams.boardSize.height <= 0) {
            std::cerr << "Board size must be positive\n";
            return false;
        }
    }
    if(parser.get<std::string>("of").find(".xml") <= 0) {
        std::cerr << "Wrong output file name: correct format is [name].xml\n";
        return false;
    }

    return true & loadFromFile(parser.get<std::string>("pf"));
}
