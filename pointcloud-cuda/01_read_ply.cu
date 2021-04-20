#include <string>
/*

struct PointCloud {
    utility::device_vector<Eigen::Vector3f> points_;
};

namespace ply_pointcloud_reader {

    struct PLYReaderState {
        utility::ConsoleProgressBar *progress_bar;
        HostPointCloud *pointcloud_ptr;
        long vertex_index;
        long vertex_num;
        long normal_index;
        long normal_num;
        long color_index;
        long color_num;
    };

    int ReadVertexCallback(p_ply_argument argument) {
        PLYReaderState *state_ptr;
        long index;
        ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                                   &index);
        if (state_ptr->vertex_index >= state_ptr->vertex_num) {
            return 0;  // some sanity check
        }

        float value = ply_get_argument_value(argument);
        state_ptr->pointcloud_ptr->points_[state_ptr->vertex_index](index) = value;
        if (index == 2) {  // reading 'z'
            state_ptr->vertex_index++;
            ++(*state_ptr->progress_bar);
        }
        return 1;
    }

    int ReadNormalCallback(p_ply_argument argument) {
        PLYReaderState *state_ptr;
        long index;
        ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                                   &index);
        if (state_ptr->normal_index >= state_ptr->normal_num) {
            return 0;
        }

        float value = ply_get_argument_value(argument);
        state_ptr->pointcloud_ptr->normals_[state_ptr->normal_index](index) = value;
        if (index == 2) {  // reading 'nz'
            state_ptr->normal_index++;
        }
        return 1;
    }

    int ReadColorCallback(p_ply_argument argument) {
        PLYReaderState *state_ptr;
        long index;
        ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                                   &index);
        if (state_ptr->color_index >= state_ptr->color_num) {
            return 0;
        }

        float value = ply_get_argument_value(argument);
        state_ptr->pointcloud_ptr->colors_[state_ptr->color_index](index) =
                value / 255.0;
        if (index == 2) {  // reading 'blue'
            state_ptr->color_index++;
        }
        return 1;
    }

}  // namespace ply_pointcloud_reader

bool ReadPointCloudFromPLY(const std::string &filename,
                           PointCloud &pointcloud,
                           bool print_progress = false);

bool ReadPointCloudFromPLY(const std::string &filename,
                           geometry::PointCloud &pointcloud,
                           bool print_progress) {
    using namespace ply_pointcloud_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: %s",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    HostPointCloud host_pc;
    state.pointcloud_ptr = &host_pc;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.");
        ply_close(ply_file);
        return false;
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;

    host_pc.Clear();
    host_pc.points_.resize(state.vertex_num);
    host_pc.normals_.resize(state.normal_num);
    host_pc.colors_.resize(state.color_num);

    utility::ConsoleProgressBar progress_bar(state.vertex_num + 1,
                                             "Reading PLY: ", print_progress);
    state.progress_bar = &progress_bar;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    ++progress_bar;
    host_pc.ToDevice(pointcloud);
    return true;
}
*/
int main() {

}