#include <iostream>
#include <vector>
#include <thread>
#include <tbb/tbb.h>

#include <pstl/algorithm>
#include <pstl/execution>

#include "ch01.h"

using ImagePtr = std::shared_ptr<ch01::Image>;

void writeImage(ImagePtr image_ptr) {
  //std::cout << "\t" << image_ptr->name() + ".bmp" << std::endl;
  image_ptr->write( (image_ptr->name() + ".bmp").c_str());
}


ImagePtr applyGamma(ImagePtr image_ptr, double gamma) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_gamma",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      const ch01::Image::Pixel& p = in_rows[i][j];
      double v = 0.3*p.bgra[2] + 0.59*p.bgra[1] + 0.11*p.bgra[0];
      double res = pow(v, gamma);
      if(res > ch01::MAX_BGR_VALUE) res = ch01::MAX_BGR_VALUE;
      out_rows[i][j] = ch01::Image::Pixel(res, res, res);
    }
  }
  return output_image_ptr;
}

ImagePtr applyTint(ImagePtr image_ptr, const double *tints) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_tinted",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  for ( int i = 0; i < height; ++i ) {
    for ( int j = 0; j < width; ++j ) {
      const ch01::Image::Pixel& p = in_rows[i][j];
      std::uint8_t b = (double)p.bgra[0] +
                       (ch01::MAX_BGR_VALUE-p.bgra[0])*tints[0];
      std::uint8_t g = (double)p.bgra[1] +
                       (ch01::MAX_BGR_VALUE-p.bgra[1])*tints[1];
      std::uint8_t r = (double)p.bgra[2] +
                       (ch01::MAX_BGR_VALUE-p.bgra[2])*tints[2];
      out_rows[i][j] =
        ch01::Image::Pixel(
          (b > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : b,
          (g > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : g,
          (r > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : r
        );
    }
  }
  return output_image_ptr;
}

void fig_1_7(const std::vector<ImagePtr>& image_vector) {
  const double tint_array[] = {0.75, 0, 0};
  for (ImagePtr img : image_vector) {
    img = applyGamma(img, 1.4);
    img = applyTint(img, tint_array);
    writeImage(img);
  }
}

void fig_1_10(const std::vector<ImagePtr> &image_vector) {
  const double tint_array[] = {0.75, 0, 0};

  tbb::flow::graph g;

  int i = 0;
  tbb::flow::source_node<ImagePtr> src(
    g,
    [&i, &image_vector](ImagePtr &out) -> bool {
      if (i < image_vector.size()) {
        out = image_vector[i++];
        return true;
      } else {
        return false;
      }
    }, false);

  tbb::flow::function_node<ImagePtr, ImagePtr> gamma(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) -> ImagePtr {
      std::cout << std::this_thread::get_id() << " gamma" << std::endl;
      return applyGamma(img, 1.4);
    }
  );

  tbb::flow::function_node<ImagePtr, ImagePtr> tint(
    g,
    tbb::flow::unlimited,
    [tint_array](ImagePtr img) -> ImagePtr {
      std::cout << std::this_thread::get_id() << " tint" << std::endl;
      return applyTint(img, tint_array);
    }
  );

  tbb::flow::function_node<ImagePtr> write(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) {
      writeImage(img);
    }
  );

  tbb::flow::make_edge(src, gamma);
  tbb::flow::make_edge(gamma, tint);
  tbb::flow::make_edge(tint, write);

  src.activate();
  g.wait_for_all();
}

ImagePtr applyGamma_parallel(ImagePtr image_ptr, double gamma) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_gamma",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  tbb::parallel_for(
    0, height,
    [&in_rows, &out_rows, width, gamma](int i) {
      for (int j = 0; j < width; ++j) {
        const ch01::Image::Pixel &p = in_rows[i][j];
        double v = 0.3 * p.bgra[2] + 0.59 * p.bgra[1] + 0.11 * p.bgra[0];
        double res = pow(v, gamma);
        if (res > ch01::MAX_BGR_VALUE) res = ch01::MAX_BGR_VALUE;
        out_rows[i][j] = ch01::Image::Pixel(res, res, res);
      }
    }
  );
  return output_image_ptr;
}

ImagePtr applyTint_parallel(ImagePtr image_ptr, const double *tints) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_tinted",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  tbb::parallel_for(
    0, height,
    [&in_rows, &out_rows, width, tints](int i) {
      for (int j = 0; j < width; ++j) {
        const ch01::Image::Pixel &p = in_rows[i][j];
        std::uint8_t b = (double) p.bgra[0] +
                         (ch01::MAX_BGR_VALUE - p.bgra[0]) * tints[0];
        std::uint8_t g = (double) p.bgra[1] +
                         (ch01::MAX_BGR_VALUE - p.bgra[1]) * tints[1];
        std::uint8_t r = (double) p.bgra[2] +
                         (ch01::MAX_BGR_VALUE - p.bgra[2]) * tints[2];
        out_rows[i][j] =
          ch01::Image::Pixel(
            (b > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : b,
            (g > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : g,
            (r > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : r
          );
      }
    }
  );
  return output_image_ptr;
}

void fig_1_11(std::vector<ImagePtr> &image_vector) {
  const double tint_array[] = {0.75, 0, 0};

  tbb::flow::graph g;

  int i = 0;
  tbb::flow::source_node<ImagePtr> src(
    g,
    [&i, &image_vector](ImagePtr &out) -> bool {
      if (i < image_vector.size()) {
        out = image_vector[i++];
        return true;
      } else {
        return false;
      }
    }, false);

  tbb::flow::function_node<ImagePtr, ImagePtr> gamma(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) -> ImagePtr {
      return applyGamma_parallel(img, 1.4);
    }
  );

  tbb::flow::function_node<ImagePtr, ImagePtr> tint(
    g,
    tbb::flow::unlimited,
    [tint_array](ImagePtr img) -> ImagePtr {
      return applyTint_parallel(img, tint_array);
    }
  );

  tbb::flow::function_node<ImagePtr> write(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) {
      writeImage(img);
    }
  );

  tbb::flow::make_edge(src, gamma);
  tbb::flow::make_edge(gamma, tint);
  tbb::flow::make_edge(tint, write);
  src.activate();
  g.wait_for_all();
}

ImagePtr applyGamma_parallel_unseq(ImagePtr image_ptr, double gamma) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_gamma",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  tbb::parallel_for(
    0, height,
    [&in_rows, &out_rows, width, gamma](int i) {
      auto in_row = in_rows[i];
      auto out_row = out_rows[i];
      std::transform(pstl::execution::unseq, in_row, in_row + width,
                     out_row, [gamma](const ch01::Image::Pixel &p) {
          double v = 0.3 * p.bgra[2] + 0.59 * p.bgra[1] + 0.11 * p.bgra[0];
          double res = pow(v, gamma);
          if (res > ch01::MAX_BGR_VALUE) res = ch01::MAX_BGR_VALUE;
          return ch01::Image::Pixel(res, res, res);
        });
    }
  );
  return output_image_ptr;
}

ImagePtr applyTint_parallel_unseq(ImagePtr image_ptr, const double *tints) {
  auto output_image_ptr =
    std::make_shared<ch01::Image>(image_ptr->name() + "_tinted",
                                  ch01::IMAGE_WIDTH, ch01::IMAGE_HEIGHT);
  auto in_rows = image_ptr->rows();
  auto out_rows = output_image_ptr->rows();
  const int height = in_rows.size();
  const int width = in_rows[1] - in_rows[0];

  tbb::parallel_for(
    0, height,
    [&in_rows, &out_rows, width, tints](int i) {
      auto in_row = in_rows[i];
      auto out_row = out_rows[i];
      std::transform(pstl::execution::unseq, in_row, in_row + width,
                     out_row, [tints](const ch01::Image::Pixel &p) {
          std::uint8_t b = (double) p.bgra[0] +(ch01::MAX_BGR_VALUE - p.bgra[0]) * tints[0];
          std::uint8_t g = (double) p.bgra[1] +(ch01::MAX_BGR_VALUE - p.bgra[1]) * tints[1];
          std::uint8_t r = (double) p.bgra[2] +(ch01::MAX_BGR_VALUE - p.bgra[2]) * tints[2];
          return ch01::Image::Pixel(
            (b > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : b,
            (g > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : g,
            (r > ch01::MAX_BGR_VALUE) ? ch01::MAX_BGR_VALUE : r
          );
        });
    }
  );
  return output_image_ptr;
}

void fig_1_12(std::vector<ImagePtr> &image_vector) {
  const double tint_array[] = {0.75, 0, 0};

  tbb::flow::graph g;

  int i = 0;
  tbb::flow::source_node<ImagePtr> src(
    g,
    [&i, &image_vector](ImagePtr &out) -> bool {
      if (i < image_vector.size()) {
        out = image_vector[i++];
        return true;
      } else {
        return false;
      }
    }, false);

  tbb::flow::function_node<ImagePtr, ImagePtr> gamma(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) -> ImagePtr {
      return applyGamma_parallel_unseq(img, 1.4);
    }
  );

  tbb::flow::function_node<ImagePtr, ImagePtr> tint(
    g,
    tbb::flow::unlimited,
    [tint_array](ImagePtr img) -> ImagePtr {
      return applyTint_parallel_unseq(img, tint_array);
    }
  );

  tbb::flow::function_node<ImagePtr> write(
    g,
    tbb::flow::unlimited,
    [](ImagePtr img) {
      writeImage(img);
    }
  );

  tbb::flow::make_edge(src, gamma);
  tbb::flow::make_edge(gamma, tint);
  tbb::flow::make_edge(tint, write);
  src.activate();
  g.wait_for_all();
}

int main(int argc, char* argv[]) {
  using namespace std::chrono_literals;
  std::vector<int> nums{2, 20, 200, 2000, 20000, 20000, 200000, 2000000, 20000000, 200000000,
                        2, 20, 200, 2000, 20000, 20000, 200000, 2000000, 20000000, 200000000,
                        2, 20, 200, 2000, 20000, 20000, 200000, 2000000, 20000000, 200000000,
                        2, 20, 200, 2000, 20000, 20000, 200000, 2000000, 20000000, 200000000};

  std::vector<ImagePtr> image_vector;
  std::for_each(pstl::execution::par, nums.begin(), nums.end(), [&image_vector](int i){
    image_vector.push_back(ch01::makeFractalImage(i));
    std::cout << std::this_thread::get_id() << " " << i << std::endl;
  });

  std::cout << "warmup" << std::endl;

  // warmup the scheduler
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });
  std::this_thread::sleep_for(3s);

  tbb::tick_count t0 = tbb::tick_count::now();
  fig_1_7(image_vector);
  std::cout << "=== plain loop Time : " << (tbb::tick_count::now()-t0).seconds() << " seconds" << std::endl;
  std::this_thread::sleep_for(3s);

  t0 = tbb::tick_count::now();
  fig_1_10(image_vector);
  std::cout << "=== flow Time : " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
  std::this_thread::sleep_for(3s);

  t0 = tbb::tick_count::now();
  fig_1_11(image_vector);
  std::cout << "=== flow parallel Time : " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
  std::this_thread::sleep_for(3s);

  t0 = tbb::tick_count::now();
  fig_1_12(image_vector);
  std::cout << "=== flow parallel unseq Time : " << (tbb::tick_count::now() - t0).seconds() << " seconds" << std::endl;
  std::this_thread::sleep_for(3s);

  return 0;
}