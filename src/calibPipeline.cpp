#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <exception>

using namespace calib;

#define CAP_DELAY 10
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 960

CalibPipeline::CalibPipeline(captureParameters params) :
    mCaptureParams(params)
{

}

PipelineExitStatus CalibPipeline::start(std::vector<Sptr<FrameProcessor>> processors)
{
    if(mCaptureParams.source == InputVideoSource::Camera && !mCapture.isOpened())
    {
        mCapture.open(mCaptureParams.camID);
        mCapture.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
        mCapture.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
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
        else if (key == 115) //s
            return PipelineExitStatus::SaveCurrentData;

        for (auto it = processors.begin(); it != processors.end(); ++it)
            if((*it)->isProcessed())
                return PipelineExitStatus::Calibrate;
    }

    return PipelineExitStatus::Calibrate;
}

cv::Size CalibPipeline::getImageSize() const
{
    return mImageSize;
}
