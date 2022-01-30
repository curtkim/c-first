#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>


using namespace gtsam;


int main(){

  // Create an empty nonlinear factor graph
  NonlinearFactorGraph graph;

  // Add a Gaussian prior on pose x_1
  Pose2 priorMean(0.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr priorNoise =
          noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  graph.add(PriorFactor<Pose2>(1, priorMean, priorNoise));

  // Add two odometry factors
  Pose2 odometry(2.0, 0.0, 0.0);
  noiseModel::Diagonal::shared_ptr odometryNoise =
          noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
  graph.add(BetweenFactor<Pose2>(1, 2, odometry, odometryNoise));
  graph.add(BetweenFactor<Pose2>(2, 3, odometry, odometryNoise));


  // create (deliberately inaccurate) initial estimate
  Values initial;
  initial.insert(1, Pose2(0.5, 0.0, 0.2));
  initial.insert(2, Pose2(2.3, 0.1, -0.2));
  initial.insert(3, Pose2(4.1, 0.1, 0.1));

  // optimize using Levenberg-Marquardt optimization
  Values result = LevenbergMarquardtOptimizer(graph, initial).optimize();

  result.print();


  // Query the marginals
  std::cout.precision(2);
  Marginals marginals(graph, result);
  std::cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << std::endl;
  std::cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << std::endl;
  std::cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << std::endl;

  return 0;
}