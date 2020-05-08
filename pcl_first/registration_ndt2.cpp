#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>

using namespace pcl;
using namespace pcl::io;

int main() {
  using PointT = PointNormal;

  PointCloud<PointT> cloud_source, cloud_target;
  loadPCDFile ("bun0.pcd", cloud_source);

  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
  transform_2.translation() << 0.1, 0.0, 0.0;
  transform_2.rotate (Eigen::AngleAxisf (M_PI/360*5, Eigen::Vector3f::UnitZ()));
  std::cout << transform_2.matrix() << std::endl;
  pcl::transformPointCloud(cloud_source, cloud_target, transform_2);

  std::cout << cloud_source.points.size() << std::endl;
  std::cout << cloud_target.points.size() << std::endl;

  PointCloud<PointT>::Ptr src = boost::make_shared<PointCloud<PointT>>(cloud_source);
  PointCloud<PointT>::Ptr tgt = boost::make_shared<PointCloud<PointT>>(cloud_target);

  PointCloud<PointT> output;

  NormalDistributionsTransform<PointT, PointT> reg;
  reg.setStepSize (0.05);
  reg.setResolution (0.025f);
  reg.setInputSource (src);
  reg.setInputTarget (tgt);
  reg.setMaximumIterations (50);
  reg.setTransformationEpsilon (1e-2);

  // Register
  reg.align (output);

  std::cout << "FinalTransformation\n" << reg.getFinalTransformation() << std::endl;
  std::cout << output.points.size() << std::endl;
  std::cout << reg.getFitnessScore () << std::endl;

  return 0;

}

