//#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/opencv.hpp>

#include <cctype>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <GL/gl.h>	// Header File For The OpenGL32 Library
#include <GL/glu.h>	// Header File For The GLu32 Library
//#include <windows.h>
#include <GL/glut.h>    // Header File For The GLUT Library
#include <unistd.h>     // Header File For sleeping.

//#include <pthread.h>
#include <iostream>


using namespace std;
using namespace cv;

const char * usage =
    " \nexample command line for calibration from a live feed.\n"
    "   calibration  -w 4 -h 5 -s 0.025 -o camera.yml -op -oe\n"
    " \n"
    " example command line for calibration from a list of stored images:\n"
    "   imagelist_creator image_list.xml *.png\n"
    "   calibration -w 4 -h 5 -s 0.025 -o camera.yml -op -oe image_list.xml\n"
    " where image_list.xml is the standard OpenCV XML/YAML\n"
    " use imagelist_creator to create the xml or yaml list\n"
    " file consisting of the list of strings, e.g.:\n"
    " \n"
    "<?xml version=\"1.0\"?>\n"
    "<opencv_storage>\n"
    "<images>\n"
    "view000.png\n"
    "view001.png\n"
    "<!-- view002.png -->\n"
    "view003.png\n"
    "view010.png\n"
    "one_extra_view.jpg\n"
    "</images>\n"
    "</opencv_storage>\n";




const char* liveCaptureHelp =
    "When the live video from camera is used as input, the following hot-keys may be used:\n"
    "  <ESC>, 'q' - quit the program\n"
    "  'g' - start capturing images\n"
    "  'u' - switch undistortion on/off\n";

static void help()
{
    printf( "This is a camera calibration sample.\n"
            "Usage: calibration\n"
            "     -w <board_width>         # the number of inner corners per one of board dimension\n"
            "     -h <board_height>        # the number of inner corners per another board dimension\n"
            "     [-pt <pattern>]          # the type of pattern: chessboard or circles' grid\n"
            "     [-n <number_of_frames>]  # the number of frames to use for calibration\n"
            "                              # (if not specified, it will be set to the number\n"
            "                              #  of board views actually available)\n"
            "     [-d <delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
            "                              # (used only for video capturing)\n"
            "     [-s <squareSize>]       # square size in some user-defined units (1 by default)\n"
            "     [-o <out_camera_params>] # the output filename for intrinsic [and extrinsic] parameters\n"
            "     [-op]                    # write detected feature points\n"
            "     [-oe]                    # write extrinsic parameters\n"
            "     [-zt]                    # assume zero tangential distortion\n"
            "     [-a <aspectRatio>]      # fix aspect ratio (fx/fy)\n"
            "     [-p]                     # fix the principal point at the center\n"
            "     [-v]                     # flip the captured images around the horizontal axis\n"
            "     [-V]                     # use a video file, and not an image list, uses\n"
            "                              # [input_data] string for the video file name\n"
            "     [-su]                    # show undistorted images after calibration\n"
            "     [input_data]             # input data, one of the following:\n"
            "                              #  - text file with a list of the images of the board\n"
            "                              #    the text file can be generated with imagelist_creator\n"
            "                              #  - name of video file with a video of the board\n"
            "                              # if input_data not specified, a live view from the camera is used\n"
            "\n" );
    printf("\n%s",usage);
    printf( "\n%s", liveCaptureHelp );
}

/* ASCII code for the escape key. */
#define ESCAPE 27

/* The number of our GLUT window */
int window;
float tx, ty, tz, rx, ry, rz = 0;
double theta = 0;


VideoCapture capture;
int i = 0;

enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
enum Pattern { CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };

vector<string> imageList;
vector<vector<Point2f> > imagePoints;
const char* outputFilename = "out_camera_data.yml";
Size boardSize, imageSize;
Pattern pattern = CHESSBOARD;
int flags = 0;
float squareSize = 1.f, aspectRatio = 1.f;
float data[3][3] = {{542, 0, 328}, {0, 541, 247}, {0, 0, 1}};
float koef[5] = {-0.28, 0.10798, -0.000556893, 0.00126, -0.027921};
Mat cameraMatrix = Mat(3, 3, CV_32FC1, &data);
Mat distCoeffs = Mat(5, 1, CV_32FC1, &koef);
bool writeExtrinsics = false, writePoints = false;
bool flipVertical = false;
std::vector<cv::Point3f> _3DPoints;
cv::Mat R(3,1,CV_64F), T(3,1,CV_64F);
clock_t prevTimestamp = 0;
int mode = CALIBRATED;
int delay = 1000;
bool undistortImage = false;
int nframes = 10;
const char* inputFilename = 0;
bool showUndistorted = false;

bool imp = true;

static double computeReprojectionErrors(
    const vector<vector<Point3f> >& objectPoints,
    const vector<vector<Point2f> >& imagePoints,
    const vector<Mat>& rvecs, const vector<Mat>& tvecs,
    const Mat& cameraMatrix, const Mat& distCoeffs,
    vector<float>& perViewErrors )
{
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f>& corners, Pattern patternType = CHESSBOARD)
{
    corners.resize(0);

    switch(patternType)
    {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for( int i = 0; i < boardSize.height; i++ )
                for( int j = 0; j < boardSize.width; j++ )
                    corners.push_back(Point3f(float(j*squareSize),
                                              float(i*squareSize), 0));
            break;

        case ASYMMETRIC_CIRCLES_GRID:
            for( int i = 0; i < boardSize.height; i++ )
                for( int j = 0; j < boardSize.width; j++ )
                    corners.push_back(Point3f(float((2*j + i % 2)*squareSize),
                                              float(i*squareSize), 0));
            break;

        default:
            CV_Error(CV_StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration( vector<vector<Point2f> > imagePoints,
                            Size imageSize, Size boardSize, Pattern patternType,
                            float squareSize, float aspectRatio,
                            int flags, Mat& cameraMatrix, Mat& distCoeffs,
                            vector<Mat>& rvecs, vector<Mat>& tvecs,
                            vector<float>& reprojErrs,
                            double& totalAvgErr)
{
    cameraMatrix = Mat::eye(3, 3, CV_64F);
    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        cameraMatrix.at<double>(0,0) = aspectRatio;

    distCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(imagePoints.size(),objectPoints[0]);

    double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
                                 distCoeffs, rvecs, tvecs, flags|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    ///*|CV_CALIB_FIX_K3*/|CV_CALIB_FIX_K4|CV_CALIB_FIX_K5);
    printf("RMS error reported by calibrateCamera: %g\n", rms);

    bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

    totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

    return ok;
}


static void saveCameraParams( const string& filename,
                              Size imageSize, Size boardSize,
                              float squareSize, float aspectRatio, int flags,
                              const Mat& cameraMatrix, const Mat& distCoeffs,
                              const vector<Mat>& rvecs, const vector<Mat>& tvecs,
                              const vector<float>& reprojErrs,
                              const vector<vector<Point2f> >& imagePoints,
                              double totalAvgErr )
{
    FileStorage fs( filename, FileStorage::WRITE );

    time_t tt;
    time( &tt );
    struct tm *t2 = localtime( &tt );
    char buf[1024];
    strftime( buf, sizeof(buf)-1, "%c", t2 );

    fs << "calibration_time" << buf;

    if( !rvecs.empty() || !reprojErrs.empty() )
        fs << "nframes" << (int)std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if( flags & CV_CALIB_FIX_ASPECT_RATIO )
        fs << "aspectRatio" << aspectRatio;

    if( flags != 0 )
    {
        sprintf( buf, "flags: %s%s%s%s",
                 flags & CV_CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                 flags & CV_CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                 flags & CV_CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                 flags & CV_CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "" );
        cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;

    fs << "avg_reprojection_error" << totalAvgErr;
    if( !reprojErrs.empty() )
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            Mat r = bigmat(Range(i, i+1), Range(0,3));
            Mat t = bigmat(Range(i, i+1), Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }

    if( !imagePoints.empty() )
    {
        Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
        for( int i = 0; i < (int)imagePoints.size(); i++ )
        {
            Mat r = imagePtMat.row(i).reshape(2, imagePtMat.cols);
            Mat imgpti(imagePoints[i]);
            imgpti.copyTo(r);
        }
        fs << "image_points" << imagePtMat;
    }
}

static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}


static bool runAndSave(const string& outputFilename,
                       const vector<vector<Point2f> >& imagePoints,
                       Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                       float aspectRatio, int flags, Mat& cameraMatrix,
                       Mat& distCoeffs, bool writeExtrinsics, bool writePoints )
{
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(imagePoints, imageSize, boardSize, patternType, squareSize,
                             aspectRatio, flags, cameraMatrix, distCoeffs,
                             rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    printf("Intrinsic paramters:\n\timage center(x,y)\t:(%f,%f)\n\tfocal length(x,y)\t:(%f,%f)\n", cameraMatrix.at<double>(0,2), cameraMatrix.at<double>(1,2), cameraMatrix.at<double>(0,0), cameraMatrix.at<double>(1,1));

    if( ok )
        saveCameraParams( outputFilename, imageSize,
                          boardSize, squareSize, aspectRatio,
                          flags, cameraMatrix, distCoeffs,
                          writeExtrinsics ? rvecs : vector<Mat>(),
                          writeExtrinsics ? tvecs : vector<Mat>(),
                          writeExtrinsics ? reprojErrs : vector<float>(),
                          writePoints ? imagePoints : vector<vector<Point2f> >(),
                          totalAvgErr );
    return ok;
}


/* A general OpenGL initialization function.  Sets all of the initial parameters. */
void InitGL(int Width, int Height)	        // We call this right after our OpenGL window is created.
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);		// This Will Clear The Background Color To Black
    glClearDepth(1.0);				// Enables Clearing Of The Depth Buffer
    glDepthFunc(GL_LESS);				// The Type Of Depth Test To Do
    glEnable(GL_DEPTH_TEST);			// Enables Depth Testing
    glShadeModel(GL_SMOOTH);			// Enables Smooth Color Shading

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();				// Reset The Projection Matrix

    gluPerspective(45.0f,(GLfloat)Width/(GLfloat)Height,0.1f,100.0f);	// Calculate The Aspect Ratio Of The Window

    glMatrixMode(GL_MODELVIEW);
}

/* The function called when our window is resized (which shouldn't happen, because we're fullscreen) */
void ReSizeGLScene(int Width, int Height)
{
    if (Height==0)				// Prevent A Divide By Zero If The Window Is Too Small
        Height=1;

    glViewport(0, 0, Width, Height);		// Reset The Current Viewport And Perspective Transformation

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    gluPerspective(45.0f,(GLfloat)Width/(GLfloat)Height,0.1f,100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void DrawRect(float x, float y)
{
    glColor3f(1.0f,0.1f,0.1f);			// set color to a blue shade.
    glBegin(GL_QUADS);				// start drawing a polygon (4 sided)
    glVertex3f(-x, y, 0.0f);		// Top Left
    glVertex3f( x, y, 0.0f);		// Top Right
    glVertex3f( x,-y, 0.0f);		// Bottom Right
    glVertex3f(-x,-y, 0.0f);		// Bottom Left
    glEnd();					// done with the polygon
}

void DrawBoard(int w, int h, int n, int m) {
    bool color = true;
    int sw = w/n, sh = h/m; //square width and height respectively
    //for each width and height draw a rectangle with a specific color
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < m; ++j) {
            //oscillate the color per square of the board
            if(color)
                glColor3f(1, 1, 1);
            else
                glColor3f(0, 0, 0);
            color = !color;

            //draw a rectangle in the ith row and jth column
            glRecti(i*sw, j*sh, (i+1)*sw, (j+1)*sh);
        }
        if(m % 2 == 0) color = !color; //switch color order at end of row if necessary
    }
}

/* The main drawing function. */
void DrawGLScene()
{
    if (imp)
    {
        Mat view, viewGray;
        bool blink = false;

        if( capture.isOpened() )
        {
            Mat view0;
            capture >> view0;
            view0.copyTo(view);
        }
        else if( i < (int)imageList.size() )
        {
            view = imread(imageList[i], 1);
            i++;
        }

        if(!view.data)
        {
            if( imagePoints.size() > 0 )
                runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints);
            imp = false;
            return;
        }

        imageSize = view.size();

        if( flipVertical )
            flip( view, view, 0 );

        vector<Point2f> pointbuf;
        cvtColor(view, viewGray, CV_BGR2GRAY);

        bool found = findChessboardCorners( view, boardSize, pointbuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

        if( mode == CAPTURING && found &&
            (!capture.isOpened() || clock() - prevTimestamp > delay*1e-3*CLOCKS_PER_SEC) )
        {
            imagePoints.push_back(pointbuf);
            prevTimestamp = clock();
            blink = capture.isOpened();
        }

        if(found)
        {
            cornerSubPix( viewGray, pointbuf, Size(11,11), Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
            drawChessboardCorners( view, boardSize, Mat(pointbuf), found );
            cv::solvePnP(cv::Mat(_3DPoints), cv::Mat(pointbuf), cameraMatrix, distCoeffs, R, T);		//Calculate the Rotation and Translation vector


            theta = sqrt((R.at<double>(0,0)*R.at<double>(0,0)) + (R.at<double>(1,0)*R.at<double>(1,0)) + (R.at<double>(2,0)*R.at<double>(2,0)));
            theta = (theta*180.0f)/3.14159f;

            //Translate the 3D chessboard
            tx = -T.at<double>(0,0);
            ty = -T.at<double>(1,0)-1.0;
            tz = -T.at<double>(2,0)+5.0;

            //Rotate the 3D chessboard about the given vector for "theta" degrees.
            rx = R.at<double>(0,0);
            ty = R.at<double>(1,0);
            rz = R.at<double>(2,0);



            cv::Mat RM(3,3,CV_64F);
            Rodrigues(R,RM);
            printf("Extrinsic parameters %d:\nR:\n",i);
            cout << RM << endl;
            printf("T:\n");
            cout << T << endl;
            printf("\n");

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		// Clear The Screen And The Depth Buffer
            glLoadIdentity();										// Reset The View

            glTranslatef(0.0f, 0.0f, -16.0f);

            //translate chessboard
            glTranslatef(tx, ty, tz);
            glRotatef(theta,rx,ry,rz);

            DrawRect(1.5, 1.5);// draw a square (quadrilateral)
            //DrawBoard(1,1,8,6);

            // swap the buffers to display, since double buffering is used.
            glutSwapBuffers();
        }

        string msg = mode == CAPTURING ? "100/100" :
                     mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(undistortImage)
                msg = format( "%d/%d Undist", (int)imagePoints.size(), nframes );
            else
                msg = format( "%d/%d", (int)imagePoints.size(), nframes );
        }

        putText( view, msg, textOrigin, 1, 1,
                 mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));

        if( blink )
            bitwise_not(view, view);

        if( mode == CALIBRATED && undistortImage )
        {
            Mat temp = view.clone();
            undistort(temp, view, cameraMatrix, distCoeffs);

        }

        imshow("Image View", view);
        int key = 0xff & waitKey(capture.isOpened() ? 50 : 500);

        if( key == 'u' && mode == CALIBRATED )
            undistortImage = !undistortImage;

        if( capture.isOpened() && key == 'g' )
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        if( mode == CAPTURING && imagePoints.size() >= (unsigned)nframes )
        {
            if( runAndSave(outputFilename, imagePoints, imageSize,
                           boardSize, pattern, squareSize, aspectRatio,
                           flags, cameraMatrix, distCoeffs,
                           writeExtrinsics, writePoints))
                mode = CALIBRATED;
            else
                mode = DETECTION;
            if( !capture.isOpened() )
            {
                imp = false;
                return;
            }
        }
    }
    else
    {
        if( !capture.isOpened() && showUndistorted )
        {
            Mat view, rview, map1, map2;
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                    getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
                                    imageSize, CV_16SC2, map1, map2);

            for( i = 0; i < (int)imageList.size(); i++ )
            {
                view = imread(imageList[i], 1);
                if(!view.data)
                    continue;
                //undistort( view, rview, cameraMatrix, distCoeffs, cameraMatrix );
                remap(view, rview, map1, map2, INTER_LINEAR);
                imshow("Image View", rview);
                int c = 0xff & waitKey();
                if( (c & 255) == 27 || c == 'q' || c == 'Q' )
                    break;
            }
        }
    }


}


/* The function called whenever a key is pressed. */
void keyPressed(unsigned char key, int x, int y)
{
    /* sleep to avoid thrashing this procedure */
    usleep(100);

    /* If escape is pressed, kill everything. */
    if (key == ESCAPE)
    {
        /* shut down our window */
        glutDestroyWindow(window);

        /* exit the program...normal termination. */
        exit(0);
    }
}

int main( int argc, char** argv )
{
    bool videofile = false;
    int cameraId = 0;


    if( argc < 2 )
    {
        help();
        return 0;
    }

    for( i = 1; i < argc; i++ )
    {
        const char* s = argv[i];
        if( strcmp( s, "-w" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &boardSize.width ) != 1 || boardSize.width <= 0 )
                return fprintf( stderr, "Invalid board width\n" ), -1;
        }
        else if( strcmp( s, "-h" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &boardSize.height ) != 1 || boardSize.height <= 0 )
                return fprintf( stderr, "Invalid board height\n" ), -1;
        }
        else if( strcmp( s, "-pt" ) == 0 )
        {
            i++;
            if( !strcmp( argv[i], "circles" ) )
                pattern = CIRCLES_GRID;
            else if( !strcmp( argv[i], "acircles" ) )
                pattern = ASYMMETRIC_CIRCLES_GRID;
            else if( !strcmp( argv[i], "chessboard" ) )
                pattern = CHESSBOARD;
            else
                return fprintf( stderr, "Invalid pattern type: must be chessboard or circles\n" ), -1;
        }
        else if( strcmp( s, "-s" ) == 0 )
        {
            if( sscanf( argv[++i], "%f", &squareSize ) != 1 || squareSize <= 0 )
                return fprintf( stderr, "Invalid board square width\n" ), -1;
        }
        else if( strcmp( s, "-n" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &nframes ) != 1 || nframes <= 3 )
                return printf("Invalid number of images\n" ), -1;
        }
        else if( strcmp( s, "-a" ) == 0 )
        {
            if( sscanf( argv[++i], "%f", &aspectRatio ) != 1 || aspectRatio <= 0 )
                return printf("Invalid aspect ratio\n" ), -1;
            flags |= CV_CALIB_FIX_ASPECT_RATIO;
        }
        else if( strcmp( s, "-d" ) == 0 )
        {
            if( sscanf( argv[++i], "%u", &delay ) != 1 || delay <= 0 )
                return printf("Invalid delay\n" ), -1;
        }
        else if( strcmp( s, "-op" ) == 0 )
        {
            writePoints = true;
        }
        else if( strcmp( s, "-oe" ) == 0 )
        {
            writeExtrinsics = true;
        }
        else if( strcmp( s, "-zt" ) == 0 )
        {
            flags |= CV_CALIB_ZERO_TANGENT_DIST;
        }
        else if( strcmp( s, "-p" ) == 0 )
        {
            flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
        }
        else if( strcmp( s, "-v" ) == 0 )
        {
            flipVertical = true;
        }
        else if( strcmp( s, "-V" ) == 0 )
        {
            videofile = true;
        }
        else if( strcmp( s, "-o" ) == 0 )
        {
            outputFilename = argv[++i];
        }
        else if( strcmp( s, "-su" ) == 0 )
        {
            showUndistorted = true;
        }
        else if( s[0] != '-' )
        {
            if( isdigit(s[0]) )
                sscanf(s, "%d", &cameraId);
            else
                inputFilename = s;
        }
        else
            return fprintf( stderr, "Unknown option %s", s ), -1;
    }


    //Initialising the 3D-Points for the chessboard
    float a = 0.2f;								//The widht/height of each square of the chessboard object
    cv::Point3f _3DPoint;
    float y = (((boardSize.height-1.0f)/2.0f)*a)+(a/2.0f);
    float x = 0.0f;
    for(int h = 0; h < boardSize.height; h++, y+=a){
        x = (((boardSize.width-2.0f)/2.0f)*(-a))-(a/2.0f);
        for(int w = 0; w < boardSize.width; w++, x+=a){
            _3DPoint.x = x;
            _3DPoint.y = y;
            _3DPoint.z = 0.0f;
            _3DPoints.push_back(_3DPoint);
        }
    }

    if( inputFilename )
    {
        if( !videofile && readStringList(inputFilename, imageList) )
            mode = CAPTURING;
        else
            capture.open(inputFilename);
    }
    else
        capture.open(cameraId);

    if( !capture.isOpened() && imageList.empty() )
        return fprintf( stderr, "Could not initialize video (%d) capture\n",cameraId ), -2;

    if( !imageList.empty() )
        nframes = (int)imageList.size();

    if( capture.isOpened() )
        printf( "%s", liveCaptureHelp );

    namedWindow( "Image View", 1 );



    char *myargv [1];
    int myargc=1;
    myargv [0]=strdup ("calibration");
    i = 0;

    /* Initialize GLUT state - glut will take any command line arguments that pertain to it or
    X Windows - look at its documentation at http://reality.sgi.com/mjk/spec3/spec3.html */
    glutInit(&myargc, myargv);

    /* Select type of Display mode:
    Double buffer
    RGBA color
    Alpha components supported
    Depth buffer */
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);

    /* get a 640 x 480 window */
    glutInitWindowSize(640, 480);

    /* the window starts at the upper left corner of the screen */
    glutInitWindowPosition(0, 0);

    /* Open a window */
    window = glutCreateWindow("Visualisasi Checkerboard");

    /* Register the function to do all our OpenGL drawing. */
    glutDisplayFunc(&DrawGLScene);

    /* Go fullscreen.  This is as soon as possible. */
    //glutFullScreen();

    /* Even if there are no events, redraw our gl scene. */
    glutIdleFunc(&DrawGLScene);

    /* Register the function called when our window is resized. */
    glutReshapeFunc(&ReSizeGLScene);

    /* Register the function called when the keyboard is pressed. */
    glutKeyboardFunc(&keyPressed);

    /* Initialize our window. */
    InitGL(640, 480);

    /* Start Event Processing Engine */
    glutMainLoop();

    return 0;
}