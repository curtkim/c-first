#include <nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>

static void wait(int seconds) {
  nvtxRangePush(__FUNCTION__);
  nvtxMark("Waiting...");
  sleep(seconds);
  nvtxRangePop();
}

int main(void) {
  nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
  sleep(1);
  nvtxRangePush(__FUNCTION__);
  wait(1);
  nvtxRangePop();
}