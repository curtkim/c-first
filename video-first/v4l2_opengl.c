#include <time.h>
#include <glad/glad.h>
// Include GLFW
#include <GLFW/glfw3.h>

#include "common/common_v4l2.h"

static unsigned long common_fps_last_time_nanos;
static unsigned long common_get_nanos(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static void common_fps_init() {
  common_fps_last_time_nanos = common_get_nanos();
}
static void common_fps_print() {
  unsigned long t;
  unsigned long dt;
  static unsigned long nframes = 0;
  nframes++;
  t = common_get_nanos();
  dt = t - common_fps_last_time_nanos;
  if (dt > 250000000) {
    printf("FPS = %f\n", (nframes / (dt / 1000000000.0)));
    common_fps_last_time_nanos = t;
    nframes = 0;
  }
}

/* Build and compile shader program, return its ID. */
GLuint common_get_shader_program(
  const char *vertex_shader_source,
  const char *fragment_shader_source
) {
  GLchar *log = NULL;
  GLint log_length, success;
  GLuint fragment_shader, program, vertex_shader;

  /* Vertex shader */
  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader);
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
  glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &log_length);
  log = malloc(log_length);
  if (log_length > 0) {
    glGetShaderInfoLog(vertex_shader, log_length, NULL, log);
    printf("vertex shader log:\n\n%s\n", log);
  }
  if (!success) {
    printf("vertex shader compile error\n");
    exit(EXIT_FAILURE);
  }

  /* Fragment shader */
  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader);
  glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
  glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &log_length);
  if (log_length > 0) {
    log = realloc(log, log_length);
    glGetShaderInfoLog(fragment_shader, log_length, NULL, log);
    printf("fragment shader log:\n\n%s\n", log);
  }
  if (!success) {
    printf("fragment shader compile error\n");
    exit(EXIT_FAILURE);
  }

  /* Link shaders */
  program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
  if (log_length > 0) {
    log = realloc(log, log_length);
    glGetProgramInfoLog(program, log_length, NULL, log);
    printf("shader link log:\n\n%s\n", log);
  }
  if (!success) {
    printf("shader link error");
    exit(EXIT_FAILURE);
  }

  free(log);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  return program;
}


static const GLuint WIDTH = 640;
static const GLuint HEIGHT = 480;
static const GLfloat vertices[] = {
/*  xy            uv */
  -1.0,  1.0,   0.0, 1.0,
  0.0,  1.0,   0.0, 0.0,
  0.0, -1.0,   1.0, 0.0,
  -1.0, -1.0,   1.0, 1.0,
};
static const GLuint indices[] = {
  0, 1, 2,
  0, 2, 3,
};

static const GLchar *vertex_shader_source =
  "#version 330 core\n"
  "in vec2 coord2d;\n"
  "in vec2 vertexUv;\n"
  "out vec2 fragmentUv;\n"
  "void main() {\n"
  "    gl_Position = vec4(coord2d, 0, 1);\n"
  "    fragmentUv = vertexUv;\n"
  "}\n";
static const GLchar *fragment_shader_source =
  "#version 330 core\n"
  "in vec2 fragmentUv;\n"
  "out vec3 color;\n"
  "uniform sampler2D myTextureSampler;\n"
  "void main() {\n"
  "    color = texture(myTextureSampler, fragmentUv.yx).rgb;\n"
  "}\n";

static const GLchar *vertex_shader_source2 =
  "#version 330 core\n"
  "in vec2 coord2d;\n"
  "in vec2 vertexUv;\n"
  "out vec2 fragmentUv;\n"
  "void main() {\n"
  "    gl_Position = vec4(coord2d + vec2(1.0, 0.0), 0, 1);\n"
  "    fragmentUv = vertexUv;\n"
  "}\n";
static const GLchar *fragment_shader_source2 =
  "#version 330 core\n"
  "in vec2 fragmentUv;\n"
  "out vec3 color;\n"
  "uniform sampler2D myTextureSampler;\n"
  "// pixel Delta. How large a pixel is in 0.0 to 1.0 that textures use.\n"
  "uniform vec2 pixD;\n"
  "void main() {\n"

  /*"// Identity\n"*/
  /*"    color = texture(myTextureSampler, fragmentUv.yx ).rgb;\n"*/

  /*"// Inverter\n"*/
  /*"    color = 1.0 - texture(myTextureSampler, fragmentUv.yx ).rgb;\n"*/

  /*"// Swapper\n"*/
  /*"    color = texture(myTextureSampler, fragmentUv.yx ).gbr;\n"*/

  /*"// Double vision ortho.\n"*/
  /*"    color = ("*/
  /*"        texture(myTextureSampler, fragmentUv.yx ).rgb +\n"*/
  /*"        texture(myTextureSampler, fragmentUv.xy ).rgb\n"*/
  /*"    ) / 2.0;\n"*/

  /*"// Multi-me.\n"*/
  /*"    color = texture(myTextureSampler, 4.0 * fragmentUv.yx ).rgb;\n"*/

  /*"// Horizontal linear blur.\n"*/
  /*"    int blur_width = 21;\n"*/
  /*"    int blur_width_half = blur_width / 2;\n"*/
  /*"    color = vec3(0.0, 0.0, 0.0);\n"*/
  /*"    for (int i = -blur_width_half; i <= blur_width_half; ++i) {\n"*/
  /*"       color += texture(myTextureSampler, vec2(fragmentUv.y + i * pixD.x, fragmentUv.x)).rgb;\n"*/
  /*"    }\n"*/
  /*"    color /= blur_width;\n"*/

  /*"// Square linear blur.\n"*/
  "    int blur_width = 23;\n"
  "    int blur_width_half = blur_width / 2;\n"
  "    color = vec3(0.0, 0.0, 0.0);\n"
  "    for (int i = -blur_width_half; i <= blur_width_half; ++i) {\n"
  "       for (int j = -blur_width_half; j <= blur_width_half; ++j) {\n"
  "           color += texture(\n"
  "               myTextureSampler, fragmentUv.yx + ivec2(i, j) * pixD\n"
  "           ).rgb;\n"
  "       }\n"
  "    }\n"
  "    color /= (blur_width * blur_width);\n"

  "}\n";

int main(int argc, char **argv) {
  CommonV4l2 common_v4l2;
  GLFWwindow *window;
  GLint
    coord2d_location,
    myTextureSampler_location,
    vertexUv_location,
    coord2d_location2,
    pixD_location2,
    myTextureSampler_location2,
    vertexUv_location2
  ;
  GLuint
    ebo,
    program,
    program2,
    texture,
    vbo,
    vao,
    vao2
  ;
  unsigned int
    cpu,
    width,
    height
  ;
  uint8_t *image;
  float *image2 = NULL;
  /*uint8_t *image2 = NULL;*/

  if (argc > 1) {
    width = strtol(argv[1], NULL, 10);
  } else {
    width = WIDTH;
  }
  if (argc > 2) {
    height = strtol(argv[2], NULL, 10);
  } else {
    height = HEIGHT;
  }
  if (argc > 3) {
    cpu = (argv[3][0] == '1');
  } else {
    cpu = 0;
  }

  /* Window system. */
  glfwInit();
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
  window = glfwCreateWindow(2 * width, height, __FILE__, NULL, NULL);
  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    printf("Failed to initialize GLAD\n");
    return -1;
  }

  CommonV4l2_init(&common_v4l2, COMMON_V4L2_DEVICE, width, height, V4L2_PIX_FMT_RGB24);

  /* Shader setup. */
  program = common_get_shader_program(vertex_shader_source, fragment_shader_source);
  coord2d_location = glGetAttribLocation(program, "coord2d");
  vertexUv_location = glGetAttribLocation(program, "vertexUv");
  myTextureSampler_location = glGetUniformLocation(program, "myTextureSampler");

  /* Shader setup 2. */
  const GLchar *fs;
  if (cpu) {
    fs = fragment_shader_source;
  } else {
    fs = fragment_shader_source2;
  }
  program2 = common_get_shader_program(vertex_shader_source2, fs);
  coord2d_location2 = glGetAttribLocation(program2, "coord2d");
  vertexUv_location2 = glGetAttribLocation(program2, "vertexUv");
  myTextureSampler_location2 = glGetUniformLocation(program2, "myTextureSampler");
  pixD_location2 = glGetUniformLocation(program2, "pixD");

  /* Create vbo. */
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  /* Create ebo. */
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  /* vao. */
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(coord2d_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(vertices[0]), (GLvoid*)0);
  glEnableVertexAttribArray(coord2d_location);
  glVertexAttribPointer(vertexUv_location, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(vertices[0])));
  glEnableVertexAttribArray(vertexUv_location);
  glBindVertexArray(0);

  /* vao2. */
  glGenVertexArrays(1, &vao2);
  glBindVertexArray(vao2);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(coord2d_location2, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(vertices[0]), (GLvoid*)0);
  glEnableVertexAttribArray(coord2d_location2);
  glVertexAttribPointer(vertexUv_location2, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(vertices[0])));
  glEnableVertexAttribArray(vertexUv_location2);
  glBindVertexArray(0);

  /* Texture buffer. */
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  /* Constant state. */
  glViewport(0, 0, 2 * width, height);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glActiveTexture(GL_TEXTURE0);

  /* Main loop. */
  common_fps_init();
  do {
    /* Blocks until an image is available, thus capping FPS to that.
     * 30FPS is common in cheap webcams. */
    waitByPoll(common_v4l2.fd);
    CommonV4l2_updateImage(&common_v4l2);
    image = CommonV4l2_getImage(&common_v4l2);
    glClear(GL_COLOR_BUFFER_BIT);

    /* Original. */
    glTexImage2D(
      GL_TEXTURE_2D, 0, GL_RGB, width, height,
      0, GL_RGB, GL_UNSIGNED_BYTE, image
    );
    glUseProgram(program);
    glUniform1i(myTextureSampler_location, 0);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    /* Optional CPU modification to compare with GPU shader speed.  */
    if (cpu) {
      image2 = realloc(image2, 3 * width * height * sizeof(image2[0]));
      for (unsigned int i = 0; i < height; ++i) {
        for (unsigned int j = 0; j < width; ++j) {
          size_t index = 3 * (i * width + j);

          /* Inverter. */
          /*image2[index + 0] = 1.0 - (image[index + 0] / 255.0);*/
          /*image2[index + 1] = 1.0 - (image[index + 1] / 255.0);*/
          /*image2[index + 2] = 1.0 - (image[index + 2] / 255.0);*/

          /* Swapper. */
          /*image2[index + 0] = image[index + 1] / 255.0;*/
          /*image2[index + 1] = image[index + 2] / 255.0;*/
          /*image2[index + 2] = image[index + 0] / 255.0;*/

          /* Square linear blur. */
          int blur_width = 5;
          int blur_width_half = blur_width / 2;
          int blur_width2 = (blur_width * blur_width);
          image2[index + 0] = 0.0;
          image2[index + 1] = 0.0;
          image2[index + 2] = 0.0;
          for (int k = -blur_width_half; k <= blur_width_half; ++k) {
            for (int l = -blur_width_half; l <= blur_width_half; ++l) {
              int i2 = i + k;
              int j2 = j + l;
              // Out of bounds is black. TODO: do module to match shader exactly.
              if (i2 > 0 && i2 < (int)height && j2 > 0 && j2 < (int)width) {
                unsigned int srcIndex = index + 3 * (k * width + l);
                image2[index + 0] += image[srcIndex + 0];
                image2[index + 1] += image[srcIndex + 1];
                image2[index + 2] += image[srcIndex + 2];
              }
            }
          }
          image2[index + 0] /= (blur_width2 * 255.0);
          image2[index + 1] /= (blur_width2 * 255.0);
          image2[index + 2] /= (blur_width2 * 255.0);
        }
      }
      glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, width, height,
        0, GL_RGB, GL_FLOAT, image2
      );
    }

    /* Modified. */
    glUseProgram(program2);
    glUniform1i(myTextureSampler_location2, 0);
    glUniform2f(pixD_location2, 1.0 / width, 1.0 / height);
    glBindVertexArray(vao2);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
    common_fps_print();
  } while (!glfwWindowShouldClose(window));

  /* Cleanup. */
  if (cpu) {
    free(image2);
  }
  CommonV4l2_deinit(&common_v4l2);
  glDeleteBuffers(1, &vbo);
  glDeleteVertexArrays(1, &vao);
  glDeleteTextures(1, &texture);
  glDeleteProgram(program);
  glfwTerminate();
  return EXIT_SUCCESS;
}