#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"
#include "50_camera_opengl.hpp"

using asio::ip::tcp;

unsigned int loadTexture(std::vector<char> recv_buf) {
  unsigned int texture1;
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_BGRA, GL_UNSIGNED_BYTE, recv_buf.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  return texture1;
}

int main(int argc, char* argv[])
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  GLFWwindow *window = make_window();
  auto[VAO, VBO, EBO] = load_model();
  glBindVertexArray(VAO);


  try
  {
    asio::io_context io_context;
    tcp::resolver resolver(io_context);
    tcp::resolver::query query("localhost", "8000");
    auto endpoints = resolver.resolve(query);

    tcp::socket socket(io_context, tcp::v4());
    socket.connect(*endpoints.begin());

    Shader ourShader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
    ourShader.use();



    while(!glfwWindowShouldClose(window)){
      std::array<char, 4> len_buf;
      std::size_t len_length = asio::read(socket, asio::buffer(len_buf));

      uint32_t body_len = 0;
      std::memcpy(&body_len, len_buf.data(), sizeof(uint32_t));

      std::cout << "len_buf=["
                << (int)len_buf[0] << ',' << (int)len_buf[1] << ','
                << (int)len_buf[2] << ',' << (int)len_buf[3] << ']'
                << " " << body_len
                << std::endl;

      std::vector<char> recv_buf(body_len);
      std::size_t recv_length = asio::read(socket, asio::buffer(recv_buf));
      //std::size_t recv_length = socket.receive(asio::buffer(recv_buf));
      std::cout << recv_length << std::endl;
      std::cout << "\n" << std::this_thread::get_id() << " get ended" << std::endl;

      // render
      // ------
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      unsigned int texture = loadTexture(recv_buf);
      glBindTexture(GL_TEXTURE_2D, texture);
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
      glDeleteTextures(1, &texture);

      glfwSwapBuffers(window);
      glfwPollEvents();

    }
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  glfwTerminate();

  return 0;
}