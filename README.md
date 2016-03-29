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

Build dependences: OpenCV 3.1+OpenCV_Contrib(for charuco template support)+Lapack+QT(optionally for advanced gui).

Config file for application:

<?xml version="1.0"?>
<opencv_storage>
<charuco_dict>0</charuco_dict>
<charuco_square_lenght>200</charuco_square_lenght>
<charuco_marker_size>100</charuco_marker_size>
<max_frames_num>30</max_frames_num>
<min_frames_num>10</min_frames_num>
</opencv_storage>
