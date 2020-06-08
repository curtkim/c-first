#ifndef SHADER_HPP
#define SHADER_HPP

#include <string>

GLuint LoadShadersFromString(const std::string VertexShaderCode, const std::string FragmentShaderCode);
GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path);

#endif
