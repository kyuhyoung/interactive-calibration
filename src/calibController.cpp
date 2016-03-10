#include "calibController.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>

calib::calibController::calibController() :
    mCalibData(nullptr)
{
    mCalibFlags = 0;
}

calib::calibController::calibController(Sptr<calib::calibrationData> data, int initialFlags) :
    mCalibData(data)
{
    mCalibFlags = initialFlags;
}

void calib::calibController::updateState()
{
    if (getFramesNumberState()) {
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
            if(mCalibData->distCoeffs.at<double>(2) < eps &&
                    mCalibData->distCoeffs.at<double>(3) < eps)
                mCalibFlags |= cv::CALIB_ZERO_TANGENT_DIST;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K1)) {
            const double eps = 0.005;
            if(mCalibData->distCoeffs.at<double>(0) < eps)
                mCalibFlags |= cv::CALIB_FIX_K1;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K2)) {
            const double eps = 0.005;
            if(mCalibData->distCoeffs.at<double>(1) < eps)
                mCalibFlags |= cv::CALIB_FIX_K2;
        }

        if(!(mCalibFlags & cv::CALIB_FIX_K3)) {
            const double eps = 0.005;
            if(mCalibData->distCoeffs.at<double>(4) < eps)
                mCalibFlags |= cv::CALIB_FIX_K3;
        }

    }
}

bool calib::calibController::getCommonCalibrationState() const
{
    return false;
}

bool calib::calibController::getFramesNumberState() const
{
    return std::max(mCalibData->imagePoints.size(), mCalibData->allCharucoCorners.size()) > 10;
}

bool calib::calibController::getRMSState() const
{
    return mCalibData->totalAvgErr < 0.5;
}

int calib::calibController::getNewFlags() const
{
    return mCalibFlags;
}
