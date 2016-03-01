#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP

#include <vector>

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
    int start(std::vector<Sptr<FrameProcessor>> processors);
    cv::Size getImageSize() const;
};

}

#endif
