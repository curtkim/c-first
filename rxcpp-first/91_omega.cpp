#include <vector>
#include <rxcpp/rx.hpp>


namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
}
using namespace Rx;

#include <chrono>
#include <thread>

using namespace std;




struct Measure {
    long frame;
};

struct CarState : Measure {
};

struct LidarMeasure : Measure {
};
struct RadarMeasure : Measure  {
};
struct GPSMeasure : Measure {
};
struct CameraMeasure : Measure  {
};

struct Position {
    float x;
    float y;
};

struct Obstacle {
    int id;
    Position pos;
};

struct TrafficLight {
};

struct Control {
    float acc;
    float steering;
    float brake;
};


class Route {
};
class HDMap {
};


Position localize(LidarMeasure& lidar, GPSMeasure& gps){
    return Position();
}

vector<Obstacle> perceive_obstacle(LidarMeasure& lidar, RadarMeasure& front_radar, RadarMeasure& rear_radar, vector<CameraMeasure> images, HDMap& hdmap, Position& pos){
    return vector<Obstacle>();
}

vector<TrafficLight> perceive_traffic_light(vector<CameraMeasure> images, HDMap& hdmap, Position& pos){
    return vector<TrafficLight>();
}

vector<Position> plan(Route& route, HDMap& hdmap, Position& pos, vector<Obstacle> obstacles, vector<TrafficLight> traffic_lights){
    return std::vector<Position>();
}

Control control(vector<Position> waypoints, CarState& carState){
    return Control();
}


observable<vector<GPSMeasure>> last_elements(observable<GPSMeasure> source, int size) {
    return source.scan(vector<GPSMeasure>{}, [size](vector<GPSMeasure> list, GPSMeasure i) {
        list.push_back(i);
        if( list.size() > size)
            list.erase(list.begin());
        return list;
    });
}

void print_list(std::vector<Measure> list) {
    for( auto i : list ) {
        std::cout << i.frame << " ";
    }
    std::cout << std::endl;
}

void publish(Control ctrl){

}

int main() {
    auto lidar_period = chrono::milliseconds(50);
    auto radar_period = chrono::milliseconds(20);
    auto gps_period = chrono::milliseconds(10);
    auto car_state_period = chrono::milliseconds(10);
    auto camera_period = chrono::milliseconds(30);

    auto lidar1$ = rxcpp::sources::interval(lidar_period).map([](long frame){ return LidarMeasure{frame};});
    auto lidar2$ = rxcpp::sources::interval(lidar_period).map([](long frame){ return LidarMeasure{frame};});
    auto lidar3$ = rxcpp::sources::interval(lidar_period).map([](long frame){ return LidarMeasure{frame};});
    auto lidar4$ = rxcpp::sources::interval(lidar_period).map([](long frame){ return LidarMeasure{frame};});
    auto lidar5$ = rxcpp::sources::interval(lidar_period).map([](long frame){ return LidarMeasure{frame};});

    auto gps$ = rxcpp::sources::interval(gps_period).map([](long frame){ return GPSMeasure{frame};});

    auto front_radar$ = rxcpp::sources::interval(radar_period).map([](long frame){ return RadarMeasure{frame};});
    auto rear_radar$ = rxcpp::sources::interval(radar_period).map([](long frame){ return RadarMeasure{frame};});

    auto camera1$ = rxcpp::sources::interval(camera_period).map([](long frame){ return CameraMeasure{frame};});
    auto camera2$ = rxcpp::sources::interval(camera_period).map([](long frame){ return CameraMeasure{frame};});
    auto camera3$ = rxcpp::sources::interval(camera_period).map([](long frame){ return CameraMeasure{frame};});

    auto car_state$ = rxcpp::sources::interval(car_state_period).map([](long frame){ return CarState{frame};});


    // init
    HDMap hdmap;
    Route route;

    auto result$ = lidar1$
        .with_latest_from(car_state$, last_elements(gps$, 5), lidar2$, lidar3$, lidar4$, lidar5$, front_radar$, rear_radar$, camera1$, camera2$, camera3$)
        .map( [&hdmap, &route](std::tuple<LidarMeasure, CarState, vector<GPSMeasure>, LidarMeasure, LidarMeasure, LidarMeasure, LidarMeasure, RadarMeasure, RadarMeasure, CameraMeasure, CameraMeasure, CameraMeasure> v){
            //C++17 structured binding:
            auto [ lidar1, car_state, gps_list, lidar2, lidar3, lidar4, lidar5, front_radar, rear_radar, camera1, camera2, camera3 ] = v;

            //printf("OnNext: %ld %d %d %d %d %d \n",
            //       std::this_thread::get_id(), lidar1.frame, lidar2.frame, lidar3.frame, lidar4.frame, lidar5.frame);

            GPSMeasure last_gps = gps_list.back();
            vector<CameraMeasure> cameras {camera1, camera2, camera3};
            vector<LidarMeasure> lidars {lidar1, lidar2, lidar3, lidar4, lidar5};

            Position curr_pos = localize(lidar1, last_gps);
            vector<Obstacle> obstacles = perceive_obstacle(lidar1, front_radar, rear_radar, cameras, hdmap, curr_pos);
            vector<TrafficLight> traffic_lights = perceive_traffic_light(cameras, hdmap, curr_pos);
            vector<Position> waypoints = plan(route, hdmap, curr_pos, obstacles, traffic_lights);
            Control ctrl = control(waypoints, car_state);

            return make_tuple(lidar1.frame, car_state, curr_pos, obstacles, traffic_lights, waypoints, ctrl);
        });

    result$
        .subscribe([](std::tuple<long, CarState, Position, vector<Obstacle>, vector<TrafficLight>, vector<Position>, Control> v){
            auto [ frame, car_state, curr_pos, obstacles, traffic_lights, waypoints, ctrl ] = v;
            cout << frame << endl;
            publish(ctrl);
        });

    return 0;
}