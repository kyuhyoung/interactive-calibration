#include "calibController.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
//#include <cstdio>

calib::calibController::calibController() :
    mCalibData(nullptr)
{
    mCalibFlags = 0;
}

calib::calibController::calibController(Sptr<calib::calibrationData> data, int initialFlags, bool autoTuning) :
    mCalibData(data)
{
    mCalibFlags = initialFlags;
    mNeedTuning = autoTuning;
    mConfIntervalsState = false;
}

void calib::calibController::updateState()
{
    if(mCalibData->cameraMatrix.total()) {
        const double relErrEps = 0.05;
        bool fConfState = false, cConfState = false, dConfState = true;
        if(sigmaMult*mCalibData->stdDeviations.at<double>(0) / mCalibData->cameraMatrix.at<double>(0,0) < relErrEps &&
                sigmaMult*mCalibData->stdDeviations.at<double>(1) / mCalibData->cameraMatrix.at<double>(1,1) < relErrEps)
            fConfState = true;
        if(sigmaMult*mCalibData->stdDeviations.at<double>(2) / mCalibData->cameraMatrix.at<double>(0,2) < relErrEps &&
                sigmaMult*mCalibData->stdDeviations.at<double>(3) / mCalibData->cameraMatrix.at<double>(1,2) < relErrEps)
            cConfState = true;

        for(int i=0; i < 5; i++)
            if(mCalibData->stdDeviations.at<double>(4+i) / fabs(mCalibData->distCoeffs.at<double>(i)) > 1)
                dConfState = false;
        //printf("f state %i, d state %i, c state %i\n", fConfState, dConfState, cConfState);
        mConfIntervalsState = fConfState && cConfState && dConfState;
    }

    if (getFramesNumberState() && mNeedTuning) {
        if( !(mCalibFlags & cv::CALIB_FIX_ASPECT_RATIO) &&
            mCalibData->cameraMatrix.total()) {
            double fDiff = fabs(mCalibData->cameraMatrix.at<double>(0,0) -
                                mCalibData->cameraMatrix.at<double>(1,1));

            if (fDiff < 3*mCalibData->stdDeviations.at<double>(0) &&
                    fDiff < 3*mCalibData->stdDeviations.at<double>(1)) {
                mCalibFlags |= cv::CALIB_FIX_ASPECT_RATIO;
                mCalibData->cameraMatrix.at<double>(0,0) =
                        mCalibData->cameraMatrix.at<double>(1,1);
            }
        }

        if(!(mCalibFlags & cv::CALIB_ZERO_TANGENT_DIST)) {
            const double eps = 0.005;
            if(fabs(mCalibData->distCoeffs.at<double>(2)) < eps &&
                    fabs(mCalibData->distCoeffs.at<double>(3)) < eps)
                mCalibFlags |= cv::CALIB_ZERO_TANGENT_DIST;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K1)) {
            const double eps = 0.005;
            if(fabs(mCalibData->distCoeffs.at<double>(0)) < eps)
                mCalibFlags |= cv::CALIB_FIX_K1;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K2)) {
            const double eps = 0.005;
            if(fabs(mCalibData->distCoeffs.at<double>(1)) < eps)
                mCalibFlags |= cv::CALIB_FIX_K2;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K3)) {
            const double eps = 0.005;
            if(fabs(mCalibData->distCoeffs.at<double>(4)) < eps)
                mCalibFlags |= cv::CALIB_FIX_K3;
        }

    }
}

bool calib::calibController::getCommonCalibrationState() const
{
    int rating = (int)getFramesNumberState() + (int)getConfidenceIntrervalsState() +
            (int)getRMSState();
    return rating == 3;
}

bool calib::calibController::getFramesNumberState() const
{
    return std::max(mCalibData->imagePoints.size(), mCalibData->allCharucoCorners.size()) > 10;
}

bool calib::calibController::getConfidenceIntrervalsState() const
{
    return mConfIntervalsState;
}

bool calib::calibController::getRMSState() const
{
    return mCalibData->totalAvgErr < 0.5;
}

int calib::calibController::getNewFlags() const
{
    return mCalibFlags;
}
