#include "calibPipeline.hpp"
#include <opencv2/highgui.hpp>
#include <exception>

using namespace calib;

#define CAP_DELAY 3
#define IMAGE_WIDTH 1280
#define IMAGE_HEIGHT 960

CalibPipeline::CalibPipeline(captureParameters params) :
    mCaptureParams(params)
{

}

int CalibPipeline::start(Sptr<FrameProcessor> processor)
{
    cv::VideoCapture capture;
    if(mCaptureParams.source == InputVideoSource::Camera)
    {
        capture.open(mCaptureParams.camID);
        capture.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
        capture.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
        capture.set(CV_CAP_PROP_AUTOFOCUS, 0);
    }
    else if (mCaptureParams.source == InputVideoSource::File)
        capture.open(mCaptureParams.videoFileName);
    mImageSize = cv::Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    if(!capture.isOpened())
        throw std::runtime_error("Unable to open video source");

    cv::Mat frame, processedFrame;
    while(capture.grab() && !processor->isProcessed()) {
        capture.retrieve(frame);
        if(mCaptureParams.flipVertical)
            cv::flip(frame, frame, -1);

        processedFrame = processor->processFrame(frame);
        cv::imshow(mainWindowName, processedFrame);
        int key = cv::waitKey(CAP_DELAY);
        if(key == 27)
            return -1;
    }


    return 0;
}

cv::Size CalibPipeline::getImageSize() const
{
    return mImageSize;
}
