#pragma once

namespace cc = carla::client;
namespace cs = carla::sensor;

auto from_sensor(boost::shared_ptr<cc::Sensor> pSensor) {
  auto data$ = rxcpp::sources::create<boost::shared_ptr<cs::SensorData>>(
      [pSensor](rxcpp::subscriber<boost::shared_ptr<cs::SensorData>> s){
        std::cout << std::this_thread::get_id() << " before listen " << std::endl;
        pSensor->Listen([s](auto data){
          assert(data != nullptr);
          //boost::shared_ptr<csd::Image> image = boost::static_pointer_cast<csd::Image>(data);
          //std::cout << std::this_thread::get_id() << " in callback " << image->GetFrame() << std::endl;
          s.on_next(data);
        });
        //s.on_completed();
      });;//.subscribe_on(rxcpp::synchronize_new_thread());
  return data$;
}

