#include <iostream>
#include <string>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/vlp_grabber.h>
#include <pcl/console/parse.h>

// Point Type
// pcl::PointXYZ, pcl::PointXYZI, pcl::PointXYZRGBA
typedef pcl::PointXYZI PointType;

int main( int argc, char *argv[] )
{
  // https://midas3.kitware.com/midas/folder/12978
  std::string pcap = "../../4_intersecton_w_police_car.pcap";

  // Point Cloud
  pcl::PointCloud<PointType>::ConstPtr cloud;
  pcl::VLPGrabber grabber(pcap);

  //boost::mutex mutex;
  boost::function<void( const pcl::PointCloud<PointType>::ConstPtr& )> function =
    [ &cloud ]( const pcl::PointCloud<PointType>::ConstPtr& ptr ){
      //boost::mutex::scoped_lock lock( mutex );

      std::cout << ptr->size() << std::endl;
      /* Point Cloud Processing */

      cloud = ptr;
    };

  // Register Callback Function
  boost::signals2::connection connection = grabber.registerCallback( function );

  grabber.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Stop Grabber
  grabber.stop();


  // Disconnect Callback Function
  if( connection.connected() ){
    connection.disconnect();
  }
}