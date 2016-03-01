#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP

#include "calibCommon.hpp"
#include "frameProcessor.hpp"

namespace calib
{


class CalibPipeline
{
protected:
    captureParameters mCaptureParams;
    cv::Size mImageSize;

public:
    CalibPipeline(captureParameters params);
    int start(Sptr<FrameProcessor> processor);
    cv::Size getImageSize() const;
};

}

#endif
