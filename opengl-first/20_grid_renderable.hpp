#pragma once

class GridRenderable {
private:
    std::vector<float> g_vertex_buffer_data;
    std::vector<unsigned int> indices;

    GLuint vao;
    GLuint vbo;
    GLuint ebo;

public:
    GridRenderable(std::vector<float> g_vertex_buffer_data, std::vector<unsigned int> indices) : g_vertex_buffer_data(g_vertex_buffer_data), indices(indices){
    }

    ~GridRenderable() {
      glDeleteVertexArrays(1, &vao);
      glDeleteBuffers(1, &vbo);
      glDeleteBuffers(1, &ebo);
    }

    void init() {

      glGenVertexArrays( 1, &vao );
      glBindVertexArray( vao );

      glGenBuffers( 1, &vbo );
      glBindBuffer( GL_ARRAY_BUFFER, vbo );
      glBufferData(GL_ARRAY_BUFFER, g_vertex_buffer_data.size()*sizeof(float), g_vertex_buffer_data.data(), GL_STATIC_DRAW);

      glEnableVertexAttribArray( 0 );
      glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

      glGenBuffers( 1, &ebo );
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo );
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

      glBindVertexArray(0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      std::cout << "vao=" << vao << " vbo=" << vbo << " ebo=" << ebo << std::endl;
    }

    void render() {
      glEnable(GL_DEPTH_TEST);
      glBindVertexArray(vao);
      // 연결되지 않은 선분을 그린다. 총 length/2개의 선분을 그린다.
      glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, NULL);
      glBindVertexArray(0);
      glDisable(GL_DEPTH_TEST);
    }
};
