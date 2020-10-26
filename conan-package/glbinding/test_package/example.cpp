#include <iostream>

#include <glbinding/Version.h>
#include <glbinding/Binding.h>
#include <glbinding/gl/gl.h>
#include <glbinding-aux/ContextInfo.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using namespace gl;
using namespace glbinding;

int main()
{
	if (!glfwInit())
		return 1;

	glfwDefaultWindowHints();
	GLFWwindow *window = glfwCreateWindow(640, 480, "", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glbinding::Binding::initialize([](const char *name) {return glfwGetProcAddress(name); }, false);

	std::cout << "\n"
		<< "OpenGL Version: " << aux::ContextInfo::version().toString() << "\n"
		<< "OpenGL vendor: " << aux::ContextInfo::vendor() << "\n"
		<< "OpenGL Render: " << aux::ContextInfo::renderer() << "\n";

	return 0;
}