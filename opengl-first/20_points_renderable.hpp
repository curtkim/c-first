#pragma once

class PointsRenderable {
private:
    std::vector<float> g_vertex_buffer_data;

    GLuint vao;
    GLuint vbo;

public:
    PointsRenderable(std::vector<float> g_vertex_buffer_data) : g_vertex_buffer_data(g_vertex_buffer_data){
    }

    ~PointsRenderable() {
      glDeleteVertexArrays(1, &vao);
      glDeleteBuffers(1, &vbo);
    }

    void init() {
      glGenVertexArrays( 1, &vao );
      glBindVertexArray( vao );

      glGenBuffers( 1, &vbo );
      glBindBuffer( GL_ARRAY_BUFFER, vbo );
      glBufferData(GL_ARRAY_BUFFER, g_vertex_buffer_data.size()*sizeof(float), g_vertex_buffer_data.data(), GL_STATIC_DRAW);

      std::cout << vao << " " << vbo << std::endl;
      std::cout << "g_vertex_buffer_data.size()=" << g_vertex_buffer_data.size() << std::endl;

      glEnableVertexAttribArray( 0 );
      glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

      // unbind
      glBindVertexArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, 0);

      std::cout << "vao=" << vao << " vbo=" << vbo << std::endl;
    }

    void render() {
      // 9. Draw mesh as wireframe
      //glEnable(GL_POINT_SMOOTH); // make the point circular
      glPointSize(15);      // must be added before glDrawArrays is called

      glBindVertexArray(vao);
      glDrawArrays(GL_POINTS, 0, g_vertex_buffer_data.size()/3);

      glDisable(GL_POINT_SMOOTH); // stop the smoothing to make the points circular
      //glBindVertexArray(0);
    }

};