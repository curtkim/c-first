## 목표 
- io thread는 async하게 timeline을 채운다.
- spawn process thread using one frame
- process thread로 부터 tuple<LocationResult, PerceptionResult, DecisionResult, ControlResult>를 받는다.
- vis thread에 frame과 process result를 넘긴다.
- vis thread에서 더 이상 사용되지 않는 cell들을 release한다.(index조정 and track release함 호출)

## thread
- io thread 
- process thread
- vis thread(optional, imgui window)

## io thread
- timeline관리
- sensor socket으로 부터 데이터 수집
- process end event를 wait( by eventfd?)
- frame구성 spawn process thread
- set shared data for vis thread
- release old data

- persist timeline to disk 
  image -> encoder -> packet -> file_camera1
  etc sensor -> file_sensor
  result -> file_result

## timeline
- track
- cell
- frame
- track별 release 함수

