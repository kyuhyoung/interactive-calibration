#include "frameProcessor.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace calib;

#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 960
#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5

static cv::SimpleBlobDetector::Params getDetectorParams()
{
    cv::SimpleBlobDetector::Params detectorParams;

    detectorParams.thresholdStep = 40;
    detectorParams.minThreshold = 20;
    detectorParams.maxThreshold = 500;
    detectorParams.minRepeatability = 2;
    detectorParams.minDistBetweenBlobs = 5;

    detectorParams.filterByColor = true;
    detectorParams.blobColor = 0;

    detectorParams.filterByArea = true;
    detectorParams.minArea = 5;
    detectorParams.maxArea = 5000;

    detectorParams.filterByCircularity = false;
    detectorParams.minCircularity = 0.8f;
    detectorParams.maxCircularity = std::numeric_limits<float>::max();

    detectorParams.filterByInertia = true;
    //minInertiaRatio = 0.6;
    detectorParams.minInertiaRatio = 0.1f;
    detectorParams.maxInertiaRatio = std::numeric_limits<float>::max();

    detectorParams.filterByConvexity = true;
    //minConvexity = 0.8;
    detectorParams.minConvexity = 0.8f;
    detectorParams.maxConvexity = std::numeric_limits<float>::max();

    return detectorParams;
}

FrameProcessor::~FrameProcessor()
{

}

bool CalibProcessor::detectAndParseChessboard(const cv::Mat &frame)
{
    int chessBoardFlags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    bool isTemplateFound = cv::findChessboardCorners(frame, mBoardSize, mCurrentImagePoints, chessBoardFlags);

    if (isTemplateFound) {
        cv::Mat viewGray;
        cv::cvtColor(frame, viewGray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(viewGray, mCurrentImagePoints, cv::Size(11,11),
            cv::Size(-1,-1), cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 30, 0.1 ));
        cv::drawChessboardCorners(frame, mBoardSize, cv::Mat(mCurrentImagePoints), isTemplateFound);
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseChAruco(const cv::Mat &frame)
{
    cv::Ptr<cv::aruco::Board> board = mCharucoBoard.staticCast<cv::aruco::Board>();

    std::vector<std::vector<cv::Point2f>> corners, rejected;
    std::vector<int> ids;
    cv::aruco::detectMarkers(frame, mArucoDictionary, corners, ids, cv::aruco::DetectorParameters::create(), rejected);
    cv::aruco::refineDetectedMarkers(frame, board, corners, ids, rejected);
    cv::Mat currentCharucoCorners, currentCharucoIds;
    if(ids.size() > 0)
        cv::aruco::interpolateCornersCharuco(corners, ids, frame, mCharucoBoard, currentCharucoCorners,
                                         currentCharucoIds);
    if(ids.size() > 0) cv::aruco::drawDetectedMarkers(frame, corners);

    if(currentCharucoCorners.total() > 3) {
        float centerX = 0, centerY = 0;
        for (int i = 0; i < currentCharucoCorners.size[0]; i++) {
            centerX += currentCharucoCorners.at<float>(i, 0);
            centerY += currentCharucoCorners.at<float>(i, 1);
        }
        centerX /= currentCharucoCorners.size[0];
        centerY /= currentCharucoCorners.size[0];
        //cv::circle(frame, cv::Point2f(centerX, centerY), 10, cv::Scalar(0, 255, 0), 10);
        mTemplateLocations.insert(mTemplateLocations.begin(), cv::Point2f(centerX, centerY));
        cv::aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
        mCurrentCharucoCorners = currentCharucoCorners;
        mCurrentCharucoIds = currentCharucoIds;
        return true;
    }

    return false;
}

bool CalibProcessor::detectAndParseACircles(const cv::Mat &frame)
{
    bool isTemplateFound = findCirclesGrid(frame, mBoardSize, mCurrentImagePoints, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(isTemplateFound) {
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
        cv::drawChessboardCorners(frame, mBoardSize, cv::Mat(mCurrentImagePoints), isTemplateFound);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseDualACircles(const cv::Mat &frame)
{
    std::vector<cv::Point2f> blackPointbuf;

    cv::Mat invertedView;
    cv::bitwise_not(frame, invertedView);
    bool isWhiteGridFound = cv::findCirclesGrid(frame, mBoardSize, mCurrentImagePoints, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(!isWhiteGridFound)
        return false;
    bool isBlackGridFound = cv::findCirclesGrid(invertedView, mBoardSize, blackPointbuf, cv::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);

    if(!isBlackGridFound)
    {
        mCurrentImagePoints.clear();
        return false;
    }
    cv::drawChessboardCorners(frame, mBoardSize, cv::Mat(mCurrentImagePoints), isWhiteGridFound);
    cv::drawChessboardCorners(frame, mBoardSize, cv::Mat(blackPointbuf), isBlackGridFound);
    mCurrentImagePoints.insert(mCurrentImagePoints.end(), blackPointbuf.begin(), blackPointbuf.end());
    mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);

    return true;
}

void CalibProcessor::saveFrameData()
{
    std::vector<cv::Point3f> objectPoints;
    float squareSize = 16.3f, acircleGrid2Distance = 295.f;

    switch(mBoardType)
    {
    case TemplateType::Chessboard:
        for( int i = 0; i < mBoardSize.height; ++i )
            for( int j = 0; j < mBoardSize.width; ++j )
                objectPoints.push_back(cv::Point3f(j*squareSize, i*squareSize, 0));
        mCalibdata->imagePoints.push_back(mCurrentImagePoints);
        mCalibdata->objectPoints.push_back(objectPoints);
        break;
    case TemplateType::chAruco:
        mCalibdata->allCharucoCorners.push_back(mCurrentCharucoCorners);
        mCalibdata->allCharucoIds.push_back(mCurrentCharucoIds);
        break;
    case TemplateType::AcirclesGrid:
        for( int i = 0; i < mBoardSize.height; i++ )
            for( int j = 0; j < mBoardSize.width; j++ )
                objectPoints.push_back(cv::Point3f((2*j + i % 2)*squareSize, i*squareSize, 0));
        mCalibdata->imagePoints.push_back(mCurrentImagePoints);
        mCalibdata->objectPoints.push_back(objectPoints);
        break;
    case TemplateType::DoubleAcirclesGrid:
    {
        float gridCenterX = (2*((float)mBoardSize.width - 1) + 1)*squareSize + acircleGrid2Distance/2;
        float gridCenterY = (mBoardSize.height - 1)*squareSize/2;

        //white part
        for( int i = 0; i < mBoardSize.height; i++ )
            for( int j = 0; j < mBoardSize.width; j++ )
                objectPoints.push_back(cv::Point3f(-float((2*j + i % 2)*squareSize + acircleGrid2Distance + (2*(mBoardSize.width - 1) + 1)*squareSize - gridCenterX),
                                          -float(i*squareSize) - gridCenterY, 0));
        //black part
        for( int i = 0; i < mBoardSize.height; i++ )
            for( int j = 0; j < mBoardSize.width; j++ )
                objectPoints.push_back(cv::Point3f(-float((2*j + i % 2)*squareSize - gridCenterX),
                                          -float(i*squareSize) - gridCenterY, 0));

        mCalibdata->imagePoints.push_back(mCurrentImagePoints);
        mCalibdata->objectPoints.push_back(objectPoints);
    }
        break;
    }
}

CalibProcessor::CalibProcessor(Sptr<calibrationData> data, TemplateType board, cv::Size boardSize) :
    mCalibdata(data), mBoardType(board), mBoardSize(boardSize)
{
    mCapuredFrames = 0;
    mNeededFramesNum = 1;
    mMaxTemplateOffset = sqrt(IMAGE_HEIGHT*IMAGE_HEIGHT + IMAGE_WIDTH*IMAGE_WIDTH) / 20.0;

    switch(mBoardType)
    {
    case TemplateType::chAruco:
        mArucoDictionary = cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(0));
        mCharucoBoard = cv::aruco::CharucoBoard::create(6, 8, 200, 100, mArucoDictionary);
        break;
    case TemplateType::AcirclesGrid:
        mBlobDetectorPtr = cv::SimpleBlobDetector::create();
        break;
    case TemplateType::DoubleAcirclesGrid:
        mBlobDetectorPtr = cv::SimpleBlobDetector::create(getDetectorParams());
        break;
    }
}

cv::Mat CalibProcessor::processFrame(const cv::Mat &frame)
{
    cv::Mat frameCopy;
    frame.copyTo(frameCopy);
    bool isTemplateFound = false;
    mCurrentImagePoints.clear();
    int delayBetweenCaptures = 30;

    switch(mBoardType)
    {
    case TemplateType::Chessboard:
        isTemplateFound = detectAndParseChessboard(frameCopy);
        break;
    case TemplateType::chAruco:
        isTemplateFound = detectAndParseChAruco(frameCopy);
        break;
    case TemplateType::AcirclesGrid:
        isTemplateFound = detectAndParseACircles(frameCopy);
        break;
    case TemplateType::DoubleAcirclesGrid:
        isTemplateFound = detectAndParseDualACircles(frameCopy);
        break;
    }

    if(mTemplateLocations.size() > delayBetweenCaptures)
        mTemplateLocations.pop_back();
    if(mTemplateLocations.size() == delayBetweenCaptures && isTemplateFound)
    {
        if(cv::norm(mTemplateLocations[0] - mTemplateLocations[delayBetweenCaptures - 1]) < mMaxTemplateOffset)
        {
            saveFrameData();
            int baseLine = 400;
            std::string displayMessage = cv::format("Frame # %d captured", std::max(mCalibdata->imagePoints.size(),
                                                                                    mCalibdata->allCharucoCorners.size()));
            cv::Point textOrigin(frameCopy.cols / 10, frameCopy.rows - 2*baseLine - 10);
            cv::putText(frame, displayMessage, textOrigin, 1, VIDEO_TEXT_SIZE, cv::Scalar(0,0,255), 2);
            cv::imshow(mainWindowName, frame);
            cv::waitKey(300);
            mCapuredFrames++;

            mTemplateLocations.clear();
            mTemplateLocations.resize(delayBetweenCaptures);
        }
    }

    return frameCopy;
}

bool CalibProcessor::isProcessed() const
{
    if(mCapuredFrames < mNeededFramesNum)
        return false;
    else
        return true;
}

void CalibProcessor::resetState()
{
    mCapuredFrames = 0;
    mTemplateLocations.clear();
}

CalibProcessor::~CalibProcessor()
{

}

void ShowProcessor::drawGridPoints(const cv::Mat &frame)
{
    for(auto it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
        for(auto pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt)
            cv::circle(frame, *pointIt, POINT_SIZE, cv::Scalar(0, 255, 0));

    for(auto it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
        for(int i = 0; i < (*it).size[0]; i++)
            cv::circle(frame, cv::Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
                       POINT_SIZE, cv::Scalar(0, 255, 0));
}

ShowProcessor::ShowProcessor(Sptr<calibrationData> data) :
    mCalibdata(data)
{
    mNeedUndistrot = true;
}

cv::Mat ShowProcessor::processFrame(const cv::Mat &frame)
{
    if(mCalibdata->cameraMatrix.size[0] && mCalibdata->distCoeffs.size[0]) {
        cv::Mat frameCopy;

        if (mNeedUndistrot && (mCalibdata->allCharucoIds.size() > 3 || mCalibdata->imagePoints.size() > 3)) {
            drawGridPoints(frame);
            cv::undistort(frame, frameCopy, mCalibdata->cameraMatrix, mCalibdata->distCoeffs,
                          cv::getOptimalNewCameraMatrix(mCalibdata->cameraMatrix, mCalibdata->distCoeffs, cv::Size(frame.rows, frame.cols), 1.0, cv::Size(frame.rows, frame.cols)));
            int baseLine = 400;
            cv::Size textSize = cv::getTextSize("Undistorted view", 1, VIDEO_TEXT_SIZE, 2, &baseLine);
            cv::Point textOrigin(frame.cols - 2*textSize.width - 10, frame.rows - 2*baseLine - 10);
            cv::putText(frameCopy, "Undistorted view", textOrigin, 1, VIDEO_TEXT_SIZE, cv::Scalar(0,0,255), 2);
        }
        else {
            frame.copyTo(frameCopy);
            drawGridPoints(frameCopy);
        }
        std::string displayMessage = cv::format("Fx = %d Fy = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0,0),
                                            (int)mCalibdata->cameraMatrix.at<double>(1,1), mCalibdata->totalAvgErr);
        int baseLine = 100;
        cv::Size textSize = cv::getTextSize(displayMessage, 1, VIDEO_TEXT_SIZE - 1, 2, &baseLine);
        cv::Point textOrigin = cv::Point(20, 2*textSize.height);
        cv::putText(frameCopy, displayMessage, textOrigin, 1, VIDEO_TEXT_SIZE - 1, cv::Scalar(0,0,255), 2);

        displayMessage = cv::format("DFx = %.2f DFy = %.2f", mCalibdata->stdDeviations.at<double>(0)*1.96,
                                                    mCalibdata->stdDeviations.at<double>(1)*1.96);
        cv::putText(frameCopy, displayMessage, cv::Point(20, 4*textSize.height), 1, VIDEO_TEXT_SIZE - 1, cv::Scalar(0,0,255), 2);

        return frameCopy;
    }

    return frame;
}

bool ShowProcessor::isProcessed() const
{
    return false;
}

void ShowProcessor::resetState()
{

}

void ShowProcessor::switchUndistort()
{
    mNeedUndistrot = !mNeedUndistrot;
}

ShowProcessor::~ShowProcessor()
{

}
