#pragma once

#include <coroutine>
#include <functional>
#include <memory>
#include <queue>
#include <vector>

namespace simcpp20 {
  class scheduled_event;
  class event;
  class await_event;
  using simtime = double;
  using event_ptr = std::shared_ptr<simcpp20::event>;

/**
 * Used to run a discrete-event simulation.
 *
 * To create a new instance, default-initialize the class:
 *
 *     simcpp20::simulation sim;
 */
  class simulation {
  public:
    /**
     * Create a pending event which will be processed after the given delay.
     *
     * TODO(fschuetz04): Only enable if T is a subclass of event.
     *
     * @tparam T Event class. Must be a subclass of simcpp20::event.
     * @param delay Delay after which the event will be processed.
     * @return Created event.
     */
    template <typename T = simcpp20::event>
    std::shared_ptr<T> timeout(simtime delay) {
      auto ev = event<T>();
      ev->trigger_delayed(delay);
      return ev;
    }

    /**
     * Create a pending event.
     *
     * TODO(fschuetz04): Only enable if T is a subclass of event.
     *
     * @tparam T Event class. Must be a subclass of simcpp20::event.
     * @return Created event.
     */
    template <typename T = simcpp20::event> std::shared_ptr<T> event() {
      return std::make_shared<T>(*this);
    }

    /**
     * Create a pending event which is triggered when any of the given events is
     * processed.
     *
     * If any of the given events is already processed or the list of given events
     * is empty, the created event is immediately triggered.
     *
     * @param evs List of events.
     * @return Created event.
     */
    event_ptr any_of(std::initializer_list<event_ptr> evs);

    /**
     * Create a pending event which is triggered when all of the given events are
     * processed.
     *
     * If all of the given events are already processed or the list of given
     * events if empty, the created event is immediately triggered.
     *
     * @param evs List of events.
     * @return Created event.
     */
    event_ptr all_of(std::initializer_list<event_ptr> evs);

    /**
     * Extract the next scheduled event from the event queue and process it.
     *
     * @throw When the event queue is empty.
     */
    void step();

    /**
     * Run the simulation until the event queue is empty.
     */
    void run();

    /**
     * Run the simulation until the next scheduled event is scheduled after the
     * given target time.
     *
     * @param target Target time.
     */
    void run_until(simtime target);

    /// @return Current simulation time.
    simtime now();

    /// @return Whether the event queue is empty.
    bool empty();

  private:
    /// Current simulation time.
    simtime _now = 0;

    /// Event queue.
    std::priority_queue<scheduled_event> scheduled_evs = {};

    /**
     * Schedule the given event to be processed after the given delay.
     *
     * @param delay Delay after which to process the event.
     * @param ev Event to be processed.
     */
    void schedule(simtime delay, event_ptr ev);

    // event needs access to simulation::schedule
    friend class event;
  };

/// One event in the event queue of simulation.
  class scheduled_event {
  public:
    /**
     * Construct a new scheduled event.
     *
     * TODO(fschuetz04): Add an incrementing ID to sort events scheduled at the
     * same time by insertion order.
     *
     * @param time Time at which to process the event.
     * @param ev Event to process.
     */
    scheduled_event(simtime time, event_ptr ev);

    /**
     * @param other Scheduled event to compare to.
     * @return Whether this event is scheduled before the given event.
     */
    bool operator<(const scheduled_event &other) const;

    /// @return Time at which to process the event.
    simtime time() const;

    /// @return Event to process.
    event_ptr ev();

  private:
    /// Time at which to process the event.
    simtime _time;

    /// Event to process.
    event_ptr _ev;
  };

/// State of an event.
  enum class event_state {
    /**
     * Event has not yet been triggered or processed. This is the initial state of
     * a new event.
     */
    pending,

    /**
     * Event has been triggered and will be processed at the current simulation
     * time.
     */
    triggered,

    /// Event is currently being processed or has been processed.
    processed
  };

/// Can be awaited by processes
  class event : public std::enable_shared_from_this<event> {
  public:
    /**
     * Construct a new pending event.
     *
     * @param simulation Reference to simulation instance. Required for scheduling
     * the event.
     */
    event(simulation &sim);

    /// Destroy the event and all processes waiting for it.
    ~event();

    /**
     * Schedule the event to be processed immediately.
     *
     * Set the event state to triggered.
     */
    void trigger();

    /**
     * Schedule the event to be processed after the given delay.
     *
     * @param delay Delay after which to process the event.
     */
    void trigger_delayed(simtime delay);

    /**
     * Add a callback to the event.
     *
     * The callback will be called when the event is processed. If the event is
     * already processed, the callback is not stored, as it will never be called.
     *
     * @param cb Callback to add to the event. The callback receives the event,
     * so one function can differentiate between multiple events.
     */
    void add_callback(std::function<void(event_ptr)> cb);

    /// @return Whether the event is pending.
    bool pending();

    /// @return Whether the event is triggered or processed.
    bool triggered();

    /// @return Whether the event is processed.
    bool processed();

  private:
    /// Coroutine handles of processes waiting for this event.
    std::vector<std::coroutine_handle<>> handles = {};

    /// Callbacks for this event.
    std::vector<std::function<void(event_ptr)>> cbs = {};

    /// State of the event.
    event_state state = event_state::pending;

    /// Reference to the simulation.
    simulation &sim;

    /**
     * Add a 20_coroutine handle to the event.
     *
     * The 20_coroutine will be resumed when the event is processed. If the event is
     * already processed, the handle is not stored, as the 20_coroutine will never
     * be resumed.
     *
     * @param handle Handle of the 20_coroutine to resume when the event is
     * processed.
     */
    void add_handle(std::coroutine_handle<> handle);

    /**
     * Process the event.
     *
     * Set the event state to processed. Call all callbacks for this event.
     * Resume all coroutines waiting for this event.
     */
    void process();

    // simulation needs access to event::process
    friend class simulation;

    // await_event needs access to event::add_handle
    friend class await_event;
  };

/// Awaitable to wait for an event inside a 20_coroutine.
  class await_event {
  public:
    /**
     * Construct a new await_event.
     *
     * @param ev Event to wait for.
     */
    await_event(event_ptr ev);

    /**
     * @return Whether the event is already processed and the 20_coroutine must
     * not be paused.
     */
    bool await_ready();

    /**
     * Resume the waiting 20_coroutine when the event is processed.
     *
     * @param handle Handle of the 20_coroutine. This handle is added to the awaited
     * event.
     */
    void await_suspend(std::coroutine_handle<> handle);

    /// Called when the 20_coroutine is resumed.
    void await_resume();

  private:
    /// Event to wait for.
    event_ptr ev;
  };

/// Used to simulate an actor.
  class process {
  public:
    /// Promise type for processes.
    class promise_type {
    public:
      /**
       * Construct a new promise_type.
       *
       * @tparam Args Additional arguments passed to the process function. These
       * arguments are ignored.
       * @param sim Reference to the simulation.
       */
      template <typename... Args>
      promise_type(simulation &sim, Args &&...)
        : sim(sim), proc_ev(sim.event()) {}

      /**
       * @return process instance containing an event which will be triggered when
       * the process finishes.
       */
      process get_return_object();

      /**
       * Register the process to be started immediately.
       *
       * Create an event, which is scheduled to be processed immediately. When
       * this event is processed, the process is started.
       *
       * @return Awaitable for the initial event.
       */
      await_event initial_suspend();

      /**
       * Trigger the underlying event since the process finished.
       *
       * @return Awaitable which is always ready.
       */
      std::suspend_never final_suspend() noexcept;

      /// Does nothing.
      void unhandled_exception();

      /**
       * Transform the given event to an await_event awaitable.
       *
       * Called when using co_await on an event.
       *
       * @param ev Event to wait for.
       * @return Awaitable for the given event.
       */
      await_event await_transform(event_ptr ev);

      /**
       * Transform the given process to an await_event awaitable.
       *
       * Called when using co_await on a process.
       *
       * @param proc Process to wait for.
       * @return Awaitable for the underlying event of the process, which is
       * triggered when the process finishes.
       */
      await_event await_transform(process proc);

    private:
      /// Refernece to the simulation.
      simulation &sim;

      /// Underlying event which is triggered when the process finishes.
      event_ptr proc_ev;
    };

  private:
    /// Underlying event which is triggered when the process finishes.
    event_ptr ev;

    /**
     * Construct a new process object
     *
     * @param ev Underlying event which is triggered when the process finishes.
     */
    process(event_ptr ev);
  };
} 