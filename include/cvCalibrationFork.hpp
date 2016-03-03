#ifndef CV_CALIBRATION_FORK_HPP
#define CV_CALIBRATION_FORK_HPP

#include <opencv2/core.hpp>

using namespace cv;

namespace cvfork
{
double calibrateCamera(InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray _stdDeviations,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );

double cvCalibrateCamera2( const CvMat* object_points,
                                const CvMat* image_points,
                                const CvMat* point_counts,
                                CvSize image_size,
                                CvMat* camera_matrix,
                                CvMat* distortion_coeffs,
                                CvMat* rotation_vectors CV_DEFAULT(NULL),
                                CvMat* translation_vectors CV_DEFAULT(NULL),
                                CvMat* stdDeviations_vector CV_DEFAULT(NULL),
                                int flags CV_DEFAULT(0),
                                CvTermCriteria term_crit CV_DEFAULT(cvTermCriteria(
                                    CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,30,DBL_EPSILON)) );
}

#endif
