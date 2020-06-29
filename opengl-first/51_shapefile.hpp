#pragma once

#include <shapefil.h>
#include <numeric>

auto read_shapefile(const std::string& filename)
{
  try
  {
    SHPHandle handle = SHPOpen(filename.c_str(), "rb");
    if (reinterpret_cast<long>(handle) <= 0)
    {
      throw std::string("File " + filename + " not found");
    }

    // pass 1
    int nShapeType, nEntities;
    double adfMinBound[4], adfMaxBound[4];
    SHPGetInfo(handle, &nEntities, &nShapeType, adfMinBound, adfMaxBound );
    //auto condition = [](SHPObject* psShape){ return psShape->nSHPType == SHPT_POLYGON && psShape->nParts == 1};

    //nEntities = 3;
    std::vector<GLint> counts;

    for (int i = 0; i < nEntities; i++)
    {
      SHPObject* psShape = SHPReadObject(handle, i );
      if (psShape->nParts == 1)
      {
        counts.push_back(psShape->nVertices);
      }
      SHPDestroyObject( psShape );
    }

    GLint sum = std::accumulate(counts.begin(), counts.end(), decltype(counts)::value_type(0));

    // pass 2
    std::vector<float> g_vertex_buffer_data;
    for (int i = 0; i < nEntities; i++)
    {
      SHPObject* psShape = SHPReadObject(handle, i );
      if (psShape->nParts == 1)
      {
        double* x = psShape->padfX;
        double* y = psShape->padfY;

        for (int v = 0; v < psShape->nVertices; v++)
        {
          float vx = (float)x[v];
          float vy = (float)y[v];
          g_vertex_buffer_data.push_back(vx);
          g_vertex_buffer_data.push_back(vy);
          g_vertex_buffer_data.push_back(0.0);
          //std::cout << vx << " " << vy << std::endl;
        }
      }
      SHPDestroyObject( psShape );
    }

    SHPClose(handle);

    return std::make_tuple(counts, g_vertex_buffer_data);
  }
  catch(const std::string& s)
  {
    throw s;
  }
  catch(...)
  {
    throw std::string("Other exception");
  }

}