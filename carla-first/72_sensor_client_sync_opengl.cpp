#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"
#include "51_camera_opengl.hpp"
#include "70_header.hpp"

using asio::ip::tcp;

unsigned int loadTexture(std::vector<char> recv_buf, int width, int height) {
  unsigned int texture1;
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, recv_buf.data());
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
    tcp::resolver::query query("localhost", "7000");
    auto endpoints = resolver.resolve(query);

    tcp::socket socket(io_context, tcp::v4());
    socket.connect(*endpoints.begin());

    Shader ourShader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
    ourShader.use();



    while(!glfwWindowShouldClose(window)){
      Header header;
      std::size_t len_length = asio::read(socket, asio::buffer(&header, sizeof(header)));

      std::vector<char> topic_name_buf(header.topic_name_length);
      std::size_t topic_name_length = asio::read(socket, asio::buffer(topic_name_buf));
      std::string topic_name(topic_name_buf.begin(), topic_name_buf.end());
      std::cout << header << std::endl;
      std::cout << "topic= " << topic_name << std::endl;

      std::vector<char> body_buf(header.body_length);
      std::size_t recv_length = asio::read(socket, asio::buffer(body_buf));
      std::cout << recv_length << std::endl;
      std::cout << "\n" << std::this_thread::get_id() << " get ended" << std::endl;

      if(header.record_type != 0 || topic_name != "/camera/1")
        continue;

      std::cout << header.body_length << " " << header.param1 << " " << header.param2 << std::endl;
      assert(header.body_length == header.param1*header.param2*4);

      // render
      // ------
      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      unsigned int texture = loadTexture(body_buf, header.param1, header.param2);
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