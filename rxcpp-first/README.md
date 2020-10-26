## connectable_observable
a source of values that is shared across all subscribers
and does not start until connectable_observable::connect() is called.
connectable_observable의 반대는 무엇인가?(connect를 호출하지 않아도 start하는)

create로 생성할때 모두 publish, connect가 필요한가?
connectable_observable와 hot / cold 개념은 다른 것인가?

# Scheduler, Worker, Schedulable, Coordination
https://stackoverflow.com/questions/30292079/schedulers-in-rxcpp

## Scheduler
scheduler owns a timeline that is exposed by the now() method.
scheduler is also a factory for workers in that timeline.
since a scheduler owns a timeline it is possible to build schedulers that time-travel.
the virtual-scheduler is a base for the test-scheduler that uses this to make multi-second tests complete in ms.

- make_event_loop
- make_new_thread
- make_same_worker
- make_run_loop
- make_immediate
- make_current_thread

- virtual clock_type::time_point now() const
- virtual void schedule(const schedulable& scbl) const
- virtual void schedule(clock_type::time_point when, const schedulable& scbl) const

## Worker
worker owns a queue of pending schedulables for the timeline and has a lifetime
The queue maintains insertion order so that when N schedulables have the same target time they are run in the order that they were inserted into the queue
The worker guarantees that each schedulable completes before the next schedulable is started
when the worker's lifetime is unsubscribed 
all pending schedulables are discarded.

## Schedulable
schedulable owns a function and has a worker and a lifetime
the schedulable is passed to the function and allows the function to reschedule itself or schedule something else on the same worker.

## Coordination
coordination is a factory for coordinators and has a scheduler
- identity_... coordinations in rxcpp are used by default and have no overhead
- syncronize_... use mutex
- observe_on_... queue-onto-a-worker
to interleave multiple streams safely
All the operators that take multiple streams or deal in time (even subscribe_on and observe_on) take a coordination parameter, not scheduler.

