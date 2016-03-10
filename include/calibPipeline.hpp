#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP

#include <vector>
#include <opencv2/highgui.hpp>

#include "calibCommon.hpp"
#include "calibController.hpp"
#include "frameProcessor.hpp"

namespace calib
{

enum class PipelineExitStatus { Finished, DeleteLastFrame, Calibrate, DeleteAllFrames, SaveCurrentData, SwitchUndistort };

class CalibPipeline
{
protected:
    captureParameters mCaptureParams;
    cv::Size mImageSize;
    cv::VideoCapture mCapture;
    Sptr<calibController> mController;

public:
    CalibPipeline(captureParameters params, Sptr<calibController> controller);
    PipelineExitStatus start(std::vector<Sptr<FrameProcessor>> processors);
    cv::Size getImageSize() const;
};

}

#endif
