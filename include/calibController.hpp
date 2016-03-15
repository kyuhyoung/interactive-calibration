#ifndef CALIB_CONTROLLER_HPP
#define CALIB_CONTROLLER_HPP

#include "calibCommon.hpp"
#include <stack>

namespace calib {

    class calibController
    {
    protected:
        Sptr<calibrationData> mCalibData;
        int mCalibFlags;
        bool mNeedTuning;
        bool mConfIntervalsState;
    public:
        calibController();
        calibController(Sptr<calibrationData> data, int initialFlags, bool autoTuning);

        void updateState();

        bool getCommonCalibrationState() const;

        bool getFramesNumberState() const;
        bool getConfidenceIntrervalsState() const;
        bool getRMSState() const;
        int getNewFlags() const;
    };

    class calibDataController
    {
    protected:
        Sptr<calibrationData> mCalibData;
        std::stack<cameraParameters> mParamsStack;
    public:
        calibDataController(Sptr<calibrationData> data);
        calibDataController();
        void deleteLastFrame();
        void rememberCurrentParameters();
        void deleteAllData();
    };

}

#endif
