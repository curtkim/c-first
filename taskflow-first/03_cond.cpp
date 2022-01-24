#include <thread>
#include <taskflow/taskflow.hpp>  // Taskflow is header-only

// 실행결과
//140463088744192 init
//140463088744192 cond 0
//140463088744192 cond 1
//140463088744192 stop

//139788585768704 init
//139788585768704 cond 1
//139788585768704 stop

int main(){

  // 초기 시드 값을 시스템 클럭으로 설정한다.
  std::srand(static_cast<unsigned int>(std::time(0)));


  tf::Executor executor;
  tf::Taskflow taskflow;

  tf::Task init = taskflow.emplace([](){
    std::cout << std::this_thread::get_id() << " init\n";
  }).name("init");
  tf::Task stop = taskflow.emplace([](){
    std::cout << std::this_thread::get_id() << " stop\n";
  }).name("stop");

  // creates a condition task that returns a random binary
  tf::Task cond = taskflow.emplace([](){
    int value = std::rand() % 2;
    std::cout << std::this_thread::get_id() << " cond " << value << "\n";
    return value;
  }).name("cond");

  // creates a feedback loop {0: cond, 1: stop}
  init.precede(cond);
  cond.precede(cond, stop);  // moves on to 'cond' on returning 0, or 'stop' on 1

  executor.run(taskflow).wait();
}