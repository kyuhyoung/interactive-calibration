#ifndef CALIB_CONTROLLER_HPP
#define CALIB_CONTROLLER_HPP

#include "calibCommon.hpp"

namespace calib {

    class calibController
    {
    protected:
        Sptr<calibrationData> mCalibData;
        int mCalibFlags;
    public:
        calibController();
        calibController(Sptr<calibrationData> data, int initialFlags);

        void updateState();

        bool getCommonCalibrationState() const;

        bool getFramesNumberState() const;
        bool getConfidenceIntrervalsState() const;
        bool getRMSState() const;
        int getNewFlags() const;
    };

}

#endif
