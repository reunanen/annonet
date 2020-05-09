/*
    This example shows how to train a semantic segmentation net using images
    annotated in the "anno" program (see https://github.com/reunanen/anno).

    Instructions:
    1. Use anno to label some data.
    2. Build the annonet_train program.
    3. Run:
       ./annonet_train /path/to/anno/data
    4. Wait while the network is being trained.
    5. Build the annonet_infer example program.
    6. Run:
       ./annonet_infer /path/to/anno/data
*/

#include "annonet.h"

#include <dlib/data_io.h>

// ----------------------------------------------------------------------------------------

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
)
{
    std::cout << std::endl << "Scanning...";

    const std::vector<dlib::file> files = dlib::get_files_in_directory_tree(anno_data_folder,
        [](const dlib::file& name) {
        if (dlib::match_ending("_mask.png")(name)) {
            return false;
        }
        if (dlib::match_ending("_result.png")(name)) {
            return false;
        }
        return dlib::match_ending(".jpeg")(name)
            || dlib::match_ending(".jpg")(name)
            || dlib::match_ending(".JPG")(name)
            || dlib::match_ending(".png")(name)
            || dlib::match_ending(".PNG")(name);
    });

    std::cout << " found " << files.size() << " candidates" << std::endl;

    std::vector<image_filenames> results;

    std::chrono::steady_clock::time_point progress_last_printed = std::chrono::steady_clock::now();

    size_t added = 0, ignored = 0;

    const auto extract_classlabel = [](const dlib::file& file) {
        const auto path_length = file.full_name().length() - file.name().length();
        const auto path = file.full_name().substr(0, path_length - 1);
        const auto prev_slash_pos = path.find_last_of("/\\");
        if (prev_slash_pos != std::string::npos) {
            return path.substr(prev_slash_pos + 1);
        }
        return std::string();
    };

    for (size_t i = 0, total = files.size(); i < total; ++i) {
        const dlib::file& name = files[i];

        image_filenames image_filenames;
        image_filenames.image_filename = name;
        image_filenames.classlabel = extract_classlabel(name);

        if (image_filenames.classlabel.empty()) {
            ++ignored;
        }
        else {
            ++added;
            results.push_back(image_filenames);
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

sample read_sample(const image_filenames& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth)
{
    sample sample;
    sample.image_filenames = image_filenames;

    try {
        dlib::load_image(sample.input_image, image_filenames.image_filename);
        sample.original_width = sample.input_image.nc();
        sample.original_height = sample.input_image.nr();

        for (unsigned long i = 0, end = anno_classes.size(); i != end; ++i) {
            const auto& anno_class = anno_classes[i];
            if (anno_class.classlabel == image_filenames.classlabel) {
                sample.classlabel = i;
                break;
            }
        }
    }
    catch (std::exception& e) {
        sample.error = e.what();
    }

    if (sample.classlabel == std::numeric_limits<unsigned long>::max()) {
        sample.error = "Unknown classlabel: " + image_filenames.classlabel;
    }

    return sample;
};

void convert_for_processing(
    const NetPimpl::input_type& full_input_image,
    NetPimpl::input_type& converted,
    int dim
)
{
#if 0
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
#endif

    const auto max_input_dim = std::max(full_input_image.nr(), full_input_image.nc());
    const auto input_dim = std::max(max_input_dim, static_cast<long>(dim)); // let's not enlarge anything

    const dlib::point center_point(
        full_input_image.nc() / 2,
        full_input_image.nr() / 2
    );

    const auto rect = dlib::centered_rect(center_point, input_dim, input_dim);

    const dlib::chip_details chip_details(rect, dlib::chip_dims(dim, dim));

    extract_image_chip(full_input_image, chip_details, converted, dlib::interpolate_nearest_neighbor());

    DLIB_CASSERT(converted.nr() == dim);
    DLIB_CASSERT(converted.nc() == dim);

#if 0
    static std::mutex mtx2;
    std::lock_guard<std::mutex> lock2(mtx2);
    dlib::save_jpeg(full_input_image, "full_input_image.jpg");
    dlib::save_jpeg(converted, "converted.jpg");
#endif
}

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