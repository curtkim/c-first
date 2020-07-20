# Example
- 01_save_image : camera 이미지를 저장한다.
- 02_save_lidar : lidar measurement를 ply로 저장한다.
- 03_multi_camera_performance : camera개수, 해상도, camera_tick(시간 해상도)를 달리하면서 carla 성능(image의 개수)을 측정한다.
- 11_rxcpp_sensor_create : carla - rxcpp 연동
- 12_rxcpp_sensor_all : camera외의 다른 sensor 연동
## headless
- 21_xodr2geojson_conver : (headless) xodr -> geojson
- 22_traffic_light_info : (headless) 
## hdd write
- 31_rxcpp_image_one_file : camera 3개, file write 성능
## opengl
- 51_camera_opengl : camera -> opengl(texture)
- 52_lidar_opengl : lidar mesuare(pointcloud) -> opengl
## without header
- 61_sensor_server : image를 4byte + buffer로 보냄
- 62_sensor_client : receive and print (by feature)
- 63_sensor_client_rx : rx
- 64_sensor_client_sync : by asio::read and opengl
## with header
- 71_sensor_server : header를 사용 
- 71_sensor_server_dummy : 성능테스트를 위해 red image를 계속 보낸다.
- 71_sensor_server_rx : multi sensor, sync로 처리, async로 처리하면 client에 문제가 생긴다.
- 72_sensor_client_dummy : while, asio::read로 처리
- 72_sensor_client_sync_opengl : asio::read, opengl
- 73_sensor_client_sync_rx : rx create, asio::read   
