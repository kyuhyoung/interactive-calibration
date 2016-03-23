#ifndef CALIB_CONTROLLER_HPP
#define CALIB_CONTROLLER_HPP

#include "calibCommon.hpp"
#include <stack>
#include <string>
#include <ostream>

namespace calib {

    class calibController
    {
    protected:
        Sptr<calibrationData> mCalibData;
        int mCalibFlags;
        bool mNeedTuning;
        bool mConfIntervalsState;
        bool mCoverageQualityState;

        double estimateCoverageQuality();
    public:
        calibController();
        calibController(Sptr<calibrationData> data, int initialFlags, bool autoTuning);

        void updateState();

        bool getCommonCalibrationState() const;

        bool getFramesNumberState() const;
        bool getConfidenceIntrervalsState() const;
        bool getRMSState() const;
        bool getPointsCoverageState() const;
        int getNewFlags() const;
    };

    class calibDataController
    {
    protected:
        Sptr<calibrationData> mCalibData;
        std::stack<cameraParameters> mParamsStack;
        std::string mParamsFileName;
    public:
        calibDataController(Sptr<calibrationData> data);
        calibDataController();

        void filterFrames();
        void setParametersFileName(const std::string& name);
        void deleteLastFrame();
        void rememberCurrentParameters();
        void deleteAllData();
        bool saveCurrentCameraParameters() const;
        void printParametersToConsole(std::ostream &output) const;
        void updateUndistortMap();
    };

}

#endif
