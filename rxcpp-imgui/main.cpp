#include <imgui.h>
#include <GLFW/glfw3.h>
#include <rxcpp/rx.hpp>

GLFWwindow *make_window();
GLFWwindow * init_imgui();
void cleanup_imgui(GLFWwindow *window);
void loop_imgui(GLFWwindow *window, rxcpp::schedulers::run_loop &rl, std::function<void(int)> sendFrame);


int main(int, char **) {
  GLFWwindow * window = init_imgui();

  rxcpp::schedulers::run_loop rl;

  rxcpp::subjects::subject<int> framebus;
  auto frame$ = framebus.get_observable();
  auto frameout = framebus.get_subscriber();
  auto sendFrame = [frameout](int frame) {
      frameout.on_next(frame);
  };

  auto interval$ = rxcpp::sources::interval(std::chrono::seconds(1));

  frame$
    .with_latest_from(interval$)
    .tap([](std::tuple<int, int> v) {
      auto [frame, second] = v;

      ImGui::Begin("Another Window");
      ImGui::SetWindowFontScale(2);
      ImGui::Text("frame = %d", frame);
      ImGui::Text("second = %d", second);
      ImGui::End();

      std::cout << std::this_thread::get_id() << " in tap" << std::endl;
    })
    .subscribe();

  loop_imgui(window, rl, sendFrame);

  cleanup_imgui(window);

  return 0;
}