#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>


int main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI> ("test_pcd.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;

    for (std::size_t i = 0; i < cloud->points.size (); ++i)
        std::cout << "    " << cloud->points[i].x
                  << " "    << cloud->points[i].y
                  << " "    << cloud->points[i].z
                  << " "    << cloud->points[i].intensity
                  << std::endl;

    pcl::io::savePCDFileBinary("test_binary.pcd", *cloud);
    pcl::io::savePCDFileASCII("test_ascii.pcd", *cloud);
    return 0;
}
