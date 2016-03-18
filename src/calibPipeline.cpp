#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <exception>

using namespace calib;

#define CAP_DELAY 10

cv::Size CalibPipeline::getCameraResolution()
{
    mCapture.set(CV_CAP_PROP_FRAME_WIDTH, 10000);
    mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, 10000);
    int w = (int)mCapture.get(CV_CAP_PROP_FRAME_WIDTH);
    int h = (int)mCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
    return cv::Size(w,h);
}

CalibPipeline::CalibPipeline(captureParameters params, Sptr<calibController> controller) :
    mCaptureParams(params), mController(controller)
{

}

PipelineExitStatus CalibPipeline::start(std::vector<Sptr<FrameProcessor>> processors)
{
    if(mCaptureParams.source == InputVideoSource::Camera && !mCapture.isOpened())
    {
        mCapture.open(mCaptureParams.camID);
        cv::Size maxRes = getCameraResolution();

        if(maxRes.width > IMAGE_MAX_WIDTH) {
            double aR = (double)maxRes.width / maxRes.height;
            mCapture.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_MAX_WIDTH);
            mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_MAX_WIDTH/aR);
        }
        else if(maxRes.height > IMAGE_MAX_HEIGHT) {
            double aR = (double)maxRes.width / maxRes.height;
            mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_MAX_HEIGHT);
            mCapture.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_MAX_HEIGHT*aR);
        }
        else {
            mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, maxRes.height);
            mCapture.set(CV_CAP_PROP_FRAME_WIDTH, maxRes.width);
        }
        mCapture.set(CV_CAP_PROP_AUTOFOCUS, 0);
    }
    else if (mCaptureParams.source == InputVideoSource::File && !mCapture.isOpened())
        mCapture.open(mCaptureParams.videoFileName);
    mImageSize = cv::Size((int)mCapture.get(CV_CAP_PROP_FRAME_WIDTH), (int)mCapture.get(CV_CAP_PROP_FRAME_HEIGHT));

    if(!mCapture.isOpened())
        throw std::runtime_error("Unable to open video source");

    cv::Mat frame, processedFrame;
    while(mCapture.grab()) {
        mCapture.retrieve(frame);
        if(mCaptureParams.flipVertical)
            cv::flip(frame, frame, -1);

        frame.copyTo(processedFrame);
        for (auto it = processors.begin(); it != processors.end(); ++it)
            processedFrame = (*it)->processFrame(processedFrame);
        cv::imshow(mainWindowName, processedFrame);
        int key = cv::waitKey(CAP_DELAY);

        if(key == 27) // esc
            return PipelineExitStatus::Finished;
        else if (key == 114) // r
            return PipelineExitStatus::DeleteLastFrame;
        else if (key == 100) // d
            return PipelineExitStatus::DeleteAllFrames;
        else if (key == 115) // s
            return PipelineExitStatus::SaveCurrentData;
        else if (key == 117) // u
            return PipelineExitStatus::SwitchUndistort;

        for (auto it = processors.begin(); it != processors.end(); ++it)
            if((*it)->isProcessed()) {
                mController->updateState();
                return PipelineExitStatus::Calibrate;
            }
    }

    return PipelineExitStatus::Finished;
}

cv::Size CalibPipeline::getImageSize() const
{
    return mImageSize;
}
