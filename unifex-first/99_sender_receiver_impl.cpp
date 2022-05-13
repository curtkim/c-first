// https://twitter.com/ericniebler/status/1524901033134022656?s=20&t=umSE2ka21CUUCVSL6Cd5Vw
#include <exception>
#include <cstdio>
#include <variant>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <sstream>
#include <utility>
#include <optional>

// For debugging
std::string get_thread_id() {
  std::stringstream sout;
  sout << "0x" << std::hex << std::this_thread::get_id();
  return sout.str();
}

// In this toy implementation, a sender can only complete with a single value, or void.
template<class Snd>
using sender_result_t = typename Snd::result_t;

template<class Snd, class Rcvr>
using connect_result_t = decltype(connect(std::declval<Snd>(), std::declval<Rcvr>()));

///////////////////////////////////////////
// just(T) sender factory
///////////////////////////////////////////
template<class T, class Rcvr>
struct just_operation {
    T value_;
    Rcvr rcvr_;

    friend void start(just_operation &self) {
      std::printf("just_operation start");
      set_value(self.rcvr_, self.value_);
    }
};

template<class T>
struct just_sender {
    using result_t = T;

    T value_;

    template<class Rcvr>
    friend just_operation<T, Rcvr> connect(just_sender self, Rcvr rcvr) {
      return {self.value_, rcvr};
    }
};

///////////////////////////////////////////
// then(Sender, Function) sender adaptor
///////////////////////////////////////////
template<class Fun, class Rcvr>
struct then_receiver {
    Fun fun_;
    Rcvr rcvr_;

    friend void set_value(then_receiver self, auto... val) {
      set_value(self.rcvr_, self.fun_(val...));
    }

    friend void set_error(then_receiver self, std::exception_ptr err) {
      set_error(self.rcvr_, err);
    }

    friend void set_stopped(then_receiver self) {
      set_stopped(self.rcvr_);
    }
};

// Handle void completions:
template<class Fun, class... Vs>
struct result : std::invoke_result<Fun, Vs...> {
};
template<class Fun>
struct result<Fun, void> : std::invoke_result<Fun> {
};

template<class PrevSnd, class Fun>
struct then_sender {
    using result_t = typename result<Fun, sender_result_t<PrevSnd>>::type;

    PrevSnd prev_;
    Fun fun_;

    template<class Rcvr>
    using state_for_ = connect_result_t<PrevSnd, then_receiver<Fun, Rcvr>>;

    template<class Rcvr>
    friend state_for_<Rcvr> connect(then_sender self, Rcvr rcvr) {
      return connect(self.prev_, then_receiver<Fun, Rcvr>{self.fun_, rcvr});
    }
};

template<class PrevSnd, class Fun>
then_sender<PrevSnd, Fun> then(PrevSnd prev, Fun fun) {
  return {prev, fun};
}

///////////////////////////////////////////
// sync_wait() sender consumer
///////////////////////////////////////////
template<class T>
struct sync_wait_receiver {
    std::condition_variable &cv_;
    std::optional<T> &value_;

    friend void set_value(sync_wait_receiver self, T val) {
      self.value_.emplace(val);
      self.cv_.notify_all();
    }

    friend void set_error(sync_wait_receiver self, std::exception_ptr err) {
      self.err_ = err;
      self.cv_.notify_all();
    }

    friend void set_stopped(sync_wait_receiver self) {
      self.cv_.notify_all();
    }
};

template<class Snd>
std::optional<sender_result_t<Snd>> sync_wait(Snd snd) {
  std::mutex mtx;
  std::unique_lock<std::mutex> lk{mtx};
  std::condition_variable cv;
  std::exception_ptr err_;
  std::optional<sender_result_t<Snd>> value;

  auto op = connect(snd, sync_wait_receiver<sender_result_t<Snd>>{cv, value});
  start(op);

  cv.wait(lk);

  if (err_)
    std::rethrow_exception(err_);

  return value;
}

struct run_loop {
private:
    struct operation_interface {
        operation_interface *next_ = nullptr;

        virtual ~operation_interface() = default;

        virtual void run() = 0;
    };

    template<class Rcvr>
    struct operation_model : operation_interface {
        Rcvr rcvr_;
        run_loop &ctx_;

        operation_model(Rcvr rcvr, run_loop &ctx)
                : rcvr_(rcvr), ctx_(ctx) {}

        void run() override {
          std::printf("Running task on thread: %s\n", get_thread_id().c_str());
          set_value(rcvr_);
        }

        friend void start(operation_model &self) {
          self.start_();
        }

        void start_() {
          ctx_.push_back_(this);
        }
    };

    operation_interface *head_ = nullptr;
    operation_interface **tail_ = &head_;
    bool finish_ = false;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread thread_;

    void push_back_(operation_interface *op) {
      std::unique_lock<std::mutex> lk(mtx_);
      *std::exchange(tail_, &op->next_) = op;
      cv_.notify_one();
    }

    operation_interface *pop_front_() {
      std::unique_lock<std::mutex> lk(mtx_);
      while (nullptr == head_) {
        if (finish_)
          return nullptr;
        cv_.wait(lk);
      }
      auto *op = std::exchange(head_, head_->next_);
      if (head_ == nullptr)
        tail_ = &head_;
      return op;
    }

    struct sender {
        using result_t = void;
        run_loop &ctx_;

        template<class Rcvr>
        friend operation_model<Rcvr> connect(sender self, Rcvr rcvr) {
          return self.connect_(rcvr);
        }

        template<class Rcvr>
        operation_model<Rcvr> connect_(Rcvr rcvr) {
          return {rcvr, ctx_};
        }
    };

    struct scheduler {
        run_loop &ctx_;

        friend sender schedule(scheduler self) {
          return {self.ctx_};
        }
    };

public:
    void run() {
      while (auto *op = pop_front_()) {
        op->run();
      }
    }

    scheduler get_scheduler() {
      return {*this};
    }

    void finish() {
      std::unique_lock<std::mutex> lk(mtx_);
      finish_ = true;
      cv_.notify_all();
    }
};

class thread_context {
private:
    run_loop loop_;
    std::thread thread_;

public:
    thread_context()
            : thread_([this] { loop_.run(); }) {}

    void finish() {
      loop_.finish();
    }

    void join() {
      thread_.join();
    }

    auto get_scheduler() {
      return loop_.get_scheduler();
    }
};

int main() {
  thread_context ctx;
  std::printf("main thread: %s\n", get_thread_id().c_str());

  //just_sender<int> first{42};
  auto first = then(schedule(ctx.get_scheduler()), [] { return 42; });
  auto next = then(first, [](int i) { return i + 1; });
  auto last = then(next, [](int i) { return i + 1; });

  int i = sync_wait(last).value();
  std::printf("result: %d\n", i);
  ctx.finish();
  ctx.join();
}
