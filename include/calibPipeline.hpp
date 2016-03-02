#ifndef CALIB_PIPELINE_HPP
#define CALIB_PIPELINE_HPP

#include <vector>

#include "calibCommon.hpp"
#include "frameProcessor.hpp"

namespace calib
{

enum class PipelineExitStatus { Finished, DeleteLastFrame, Calibrate, DeleteAllFrames, SaveCurrentData };

class CalibPipeline
{
protected:
    captureParameters mCaptureParams;
    cv::Size mImageSize;

public:
    CalibPipeline(captureParameters params);
    PipelineExitStatus start(std::vector<Sptr<FrameProcessor>> processors);
    cv::Size getImageSize() const;
};

}

#endif
