#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>


using namespace gtsam;


class UnaryFactor: public NoiseModelFactor1<Pose2> {
    double mx_, my_; ///< X and Y measurements

public:
    UnaryFactor(Key j, double x, double y, const SharedNoiseModel& model):
    NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y) {}

    Vector evaluateError(const Pose2& q,
                         boost::optional<Matrix&> H = boost::none) const
                         {
      if (H) (*H) = (Matrix(2,3)<< 1.0,0.0,0.0, 0.0,1.0,0.0).finished();
      return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
                         }
};


int main(){

  // Create an empty nonlinear factor graph
  NonlinearFactorGraph graph;

  // add unary measurement factors, like GPS, on all three poses
  noiseModel::Diagonal::shared_ptr unaryNoise =
          noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1)); // 10cm std on x,y
  graph.add(boost::make_shared<UnaryFactor>(1, 0.0, 0.0, unaryNoise));
  graph.add(boost::make_shared<UnaryFactor>(2, 2.0, 0.0, unaryNoise));
  graph.add(boost::make_shared<UnaryFactor>(3, 4.0, 0.0, unaryNoise));


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