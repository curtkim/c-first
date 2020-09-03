#include <iostream>
#include <csignal>
#include <thread>

#include <unistd.h>

using namespace std;

bool bContinue = true;

void signalHandler(int signum) {
  cout << "Interrupt signal (" << signum << ") received.\n";

// cleanup and close up stuff here
// terminate program
  bContinue = false;

  //exit(signum);
}

void f() {
  while (bContinue) {
    cout << "Going to sleep...." << endl;
    sleep(1);
  }
  cout << "thread exit" << endl;
}

int main() {
  // register signal SIGINT and signal handler
  signal(SIGINT, signalHandler);

  std::thread t(f);

  if (t.joinable() == true)
    t.join();

  cout << "main exit" << endl;
  return 0;
}