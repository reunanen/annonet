#include "annonet.h"

#include <dlib/data_io.h>

// ----------------------------------------------------------------------------------------

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
)
{
    std::cout << std::endl << "Scanning...";

    const std::vector<dlib::file> files = dlib::get_files_in_directory_tree(anno_data_folder, dlib::match_ending("_0.png"));

    std::cout << " found " << files.size() << " candidates" << std::endl;

    std::vector<image_filenames> results;

    const auto file_exists = [](const std::string& filename) {
        std::ifstream f(filename, std::ios::binary);
        return !!f;
    };

    std::chrono::steady_clock::time_point progress_last_printed = std::chrono::steady_clock::now();

    size_t added = 0, ignored = 0;

    for (size_t i = 0, total = files.size(); i < total; ++i) {
        const dlib::file& name = files[i];

        image_filenames image_filenames;
        image_filenames.input0_filename = name;

        const std::string prefix = name.full_name().substr(0, name.full_name().length() - 6);

        std::string input1_filename = prefix + "_1.png";
        std::string ground_truth_filename = prefix + "_ground-truth.png";
        
        const bool input1_file_exists = file_exists(input1_filename);
        const bool ground_truth_file_exists = file_exists(ground_truth_filename);

        if (input1_file_exists) {
            image_filenames.input1_filename = input1_filename;

            if (ground_truth_file_exists) {
                image_filenames.ground_truth_filename = ground_truth_filename;
            }

            if (ground_truth_file_exists || !require_ground_truth) {
                results.push_back(image_filenames);
                ++added;
            }
            else if (require_ground_truth) {
                ++ignored;
            }
        }
        else {
            ++ignored;
        }

        const auto now = std::chrono::steady_clock::now();
        if (i == 0 || i == total - 1 || (now - progress_last_printed) > std::chrono::milliseconds(100)) {
            std::cout
                << "\rScanned " << std::fixed << std::setprecision(2)
                << ((i + 1) * 100.0) / total << " % of " << total << " files: "
                << added << " added, " << ignored << " ignored";
            progress_last_printed = now;
        }
    }

    std::cout << std::endl;

    return results;
}

void calculate_diff(NetPimpl::input_type& input_image_stack)
{
    const auto& input0 = input_image_stack[0];
    const auto& input1 = input_image_stack[1];
    auto& diff = input_image_stack[2];
    diff.set_size(input0.nr(), input0.nc());
    for (long r = 0, nr = input0.nr(); r < nr; ++r) {
        for (long c = 0, nc = input0.nc(); c < nc; ++c) {
            const auto& i0 = input0(r, c);
            const auto& i1 = input1(r, c);
            auto& d = diff(r, c);
            
            const auto calc_diff = [](uint8_t a, uint8_t b) {
                return static_cast<uint8_t>(
                    std::max(0, std::min(255,
                        128 + static_cast<int>(a) / 2 - static_cast<int>(b) / 2
                    ))
                );
            };

            d.red   = calc_diff(i0.red,   i1.red);
            d.green = calc_diff(i0.green, i1.green);
            d.blue  = calc_diff(i0.blue,  i1.blue);
        }
    }
}

sample read_sample(const image_filenames& image_filenames, bool require_ground_truth, double downscaling_factor)
{
    sample sample;
    sample.image_filenames = image_filenames;

    sample.input_image_stack.resize(3);

    try {
        dlib::load_image(sample.input_image_stack[0], image_filenames.input0_filename);
        dlib::load_image(sample.input_image_stack[1], image_filenames.input1_filename);
        sample.original_width = sample.input_image_stack[0].nc();
        sample.original_height = sample.input_image_stack[0].nr();

        if (sample.input_image_stack[1].nr() != sample.original_height || sample.input_image_stack[1].nc() != sample.original_width) {
            sample.error = "Input image size mismatch";
        }

        dlib::resize_image(1.0 / downscaling_factor, sample.input_image_stack[0]);
        dlib::resize_image(1.0 / downscaling_factor, sample.input_image_stack[1]);

        calculate_diff(sample.input_image_stack);

        if (!image_filenames.ground_truth_filename.empty()) {
            dlib::load_image(sample.target_image, image_filenames.ground_truth_filename);

            if (sample.target_image.nr() != sample.original_height || sample.target_image.nc() != sample.original_width) {
                sample.error = "Label image size mismatch";
            }
            else {
                dlib::resize_image(1.0 / downscaling_factor, sample.target_image);
                sample.target_image /= 255.f;
            }
        }
        else if (require_ground_truth) {
            sample.error = "No ground truth available";
        }
    }
    catch (std::exception& e) {
        sample.error = e.what();
    }

    return sample;
};

void set_low_priority()
{
#ifdef _WIN32
    if (!SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS)) {
        std::cerr << "Error setting low priority" << std::endl;
    }
#else // WIN32
    // TODO
#endif // WIN32
}