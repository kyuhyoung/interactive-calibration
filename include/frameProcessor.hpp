#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include "calibCommon.hpp"
#include "calibController.hpp"

namespace calib
{
class FrameProcessor
{
protected:

public:
    virtual ~FrameProcessor();
    virtual cv::Mat processFrame(const cv::Mat& frame) = 0;
    virtual bool isProcessed() const = 0;
    virtual void resetState() = 0;
};

class CalibProcessor : public FrameProcessor
{
protected:
    Sptr<calibrationData> mCalibData;
    TemplateType mBoardType;
    cv::Size mBoardSize;
    std::vector<cv::Point2f> mTemplateLocations;
    std::vector<cv::Point2f> mCurrentImagePoints;
    cv::Mat mCurrentCharucoCorners;
    cv::Mat mCurrentCharucoIds;

    cv::Ptr<cv::SimpleBlobDetector> mBlobDetectorPtr;
    cv::Ptr<cv::aruco::Dictionary> mArucoDictionary;
    cv::Ptr<cv::aruco::CharucoBoard> mCharucoBoard;

    int mNeededFramesNum;
    unsigned mDelayBetweenCaptures;
    int mCapuredFrames;
    float mMaxTemplateOffset;
    float mSquareSize;
    float mTemplDist;

    bool detectAndParseChessboard(const cv::Mat& frame);
    bool detectAndParseChAruco(const cv::Mat& frame);
    bool detectAndParseACircles(const cv::Mat& frame);
    bool detectAndParseDualACircles(const cv::Mat& frame);
    void saveFrameData();
    void showCaptureMessage(const cv::Mat &frame, const std::string& message);
    bool checkLastFrame();

public:
    CalibProcessor(Sptr<calibrationData> data, captureParameters& capParams);
    virtual cv::Mat processFrame(const cv::Mat& frame) override;
    virtual bool isProcessed() const override;
    virtual void resetState() override;
    ~CalibProcessor();
};

enum class visualisationMode {Grid, Window};

class ShowProcessor : public FrameProcessor
{
protected:
    Sptr<calibrationData> mCalibData;
    Sptr<calibController> mController;
    TemplateType mBoardType;
    visualisationMode mVisMode;
    bool mNeedUndistort;
    double mGridViewScale;
    double mTextSize;

    void drawBoard(cv::Mat& img, cv::InputArray& points);
    void drawGridPoints(const cv::Mat& frame);
public:
    ShowProcessor(Sptr<calibrationData> data, Sptr<calibController> controller, TemplateType board);
    virtual cv::Mat processFrame(const cv::Mat& frame) override;
    virtual bool isProcessed() const override;
    virtual void resetState() override;

    void setVisualizationMode(visualisationMode mode);
    void switchVisualizationMode();
    void clearBoardsView();
    void updateBoardsView();

    void switchUndistort();
    void setUndistort(bool isEnabled);
    ~ShowProcessor();
};

}


#endif
