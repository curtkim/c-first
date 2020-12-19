#include <assert.h>
#define HFSM2_ENABLE_PLANS
#include <hfsm2/machine.hpp>

struct Context {
  bool powerOn;
};
using Config = hfsm2::Config::ContextT<Context>;
using M = hfsm2::MachineT<Config>;

#define S(s) struct s

using FSM = M::PeerRoot<
  S(Off),                               // initial top-level state
  M::Composite<S(On),                   // sub-machine region with a head state (On) and and 3 sub-states
    S(Red),                             // initial sub-state of the region
    S(Yellow),
    S(Green)
  >,
  S(Done)
>;

#undef S

struct Event {};

struct Off : FSM::State
{
  void entryGuard(FullControl& control) {        // called before state activation, use to re-route transitions
    if (control.context().powerOn)               // access shared data
      control.changeTo<On>();                    // initiate a transition into 'On' region
  }
};

struct On : FSM::State
{
  void enter(PlanControl& control) {               // called on state activation
    auto plan = control.plan();                    // access the plan for the region

    plan.change<Red, Yellow>();                    // sequence plan steps, executed when the previous state succeeds
    plan.change<Yellow, Green>();
    plan.change<Green, Yellow>();
    plan.change<Yellow, Red>();
  }

  void exit(PlanControl& /*control*/) {}             // called on state deactivation

  void planSucceeded(FullControl& control) {         // called on the successful completion of all plan steps
    control.changeTo<Done>();
  }

  void planFailed(FullControl& /*control*/) {}       // called if any of the plan steps fails
};

struct Red : FSM::State
{
  void update(FullControl& control) {                // called on periodic state machine updates
    control.succeed();                             // notify successful completion of the plan step
  }                                                  // plan will advance to the 'Yellow' state
};

struct Yellow : FSM::State
{
  void update(FullControl& control) {
    control.succeed();                             // plan will advance to the 'Green' state on the first entry
    // and 'Red' state on the second one
  }
};

struct Green : FSM::State
{
  void react(const Event&, FullControl& control) {   // called on external events
    control.succeed();                             // advance to the next plan step
  }
};

struct Done : FSM::State
{};

int main() {
  Context context;
  context.powerOn = true;

  FSM::Instance fsm{context};
  assert(fsm.isActive<On>());                        // activated by Off::entryGuard()
  assert(fsm.isActive<Red>());                       // On's initial sub-state

  fsm.update();
  assert(fsm.isActive<Yellow>());                    // 1st setp of On's plan

  fsm.update();
  assert(fsm.isActive<Green>());                     // 2nd setp of On's plan

  fsm.react(Event{});
  assert(fsm.isActive<Yellow>());                    // 3rd setp of On's plan

  fsm.update();
  assert(fsm.isActive<Red>());                       // 4th setp of On's plan

  fsm.update();
  assert(fsm.isActive<Done>());                      // activated by On::planSucceeded()

  return 0;
}