# interactive-calibration
A new calibration sample for OpenCV 3.2 release

The application is designed to simplify calibration process.
Main features:
- Interactive calibration process: after each new frame user can see current results
- All boards views is dislpayed on main screen or in small window
- Auto tweaking of calibration flags
- Bad board views filtering
- Auto capturing of static boards
- Multicriterial evaluation of calibration quality 

Build dependencies: OpenCV 3.1+OpenCV_Contrib(for charuco template support)+Lapack(optionally for fast processing)+QT(optionally for advanced gui).

CMake build options:
- USE_LAPACK enables or disables Lapack
