/*
    This example shows how to train a semantic segmentation net using images
    annotated in the "anno" program (see https://github.com/reunanen/anno).

    Instructions:
    1. Use anno to label some data (use the "things" mode).
    2. Build the annonet_train program.
    3. Run:
       ./annonet_train /path/to/anno/data
    4. Wait while the network is being trained.
    5. Build the annonet_infer example program.
    6. Run:
       ./annonet_infer /path/to/anno/data
*/

#include "annonet.h"

#include "cpp-read-file-in-memory/read-file-in-memory.h"

#include <dlib/data_io.h>
#include <rapidjson/document.h>

// ----------------------------------------------------------------------------------------

bool equal_ignoring_alpha(const dlib::rgb_alpha_pixel& a, const dlib::rgb_alpha_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue;
}

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes)
{
    if (rgba_label == rgba_ignore_label) {
        return dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    }
    if (rgba_label == dlib::rgb_alpha_pixel(0, 255, 0, 64)) {
        return 0; // clean
    }
    for (const AnnoClass& anno_class : anno_classes) {
        if (equal_ignoring_alpha(anno_class.rgba_label, rgba_label)) {
            return anno_class.index;
        }
    }
    std::ostringstream error;
    error << "Unknown class: "
        << "r = " << static_cast<int>(rgba_label.red) << ", "
        << "g = " << static_cast<int>(rgba_label.green) << ", "
        << "b = " << static_cast<int>(rgba_label.blue) << ", "
        << "alpha = " << static_cast<int>(rgba_label.alpha);
    throw std::runtime_error(error.str());
}

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

    const auto file_exists = [](const std::string& filename) {
        std::ifstream label_file(filename, std::ios::binary);
        return !!label_file;
    };

    std::chrono::steady_clock::time_point progress_last_printed = std::chrono::steady_clock::now();

    size_t added = 0, ignored = 0;

    for (size_t i = 0, total = files.size(); i < total; ++i) {
        const dlib::file& name = files[i];

        image_filenames image_filenames;
        image_filenames.image_filename = name;

        const std::string label_filename = name.full_name() + "_annotation_paths.json";
        const bool label_file_exists = file_exists(label_filename);

        if (label_file_exists) {
            image_filenames.label_filename = label_filename;
        }

        const std::string segmentation_label_filename = name.full_name() + "_mask.png";
        const bool segmentation_label_file_exists = file_exists(segmentation_label_filename);

        if (segmentation_label_file_exists) {
            image_filenames.segmentation_label_filename = segmentation_label_filename;
        }

        if (label_file_exists || !require_ground_truth) {
            results.push_back(image_filenames);
            ++added;
        }
        else if (require_ground_truth) {
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

std::vector<dlib::mmod_rect> parse_labels(const std::string& json, const std::vector<AnnoClass>& anno_classes)
{
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        throw std::runtime_error("Error parsing json\n" + json);
    }

    if (!doc.IsArray()) {
        throw std::runtime_error("Unexpected annotation paths json content - the document should be an array");
    }

    std::vector<dlib::mmod_rect> mmod_rects;

    for (rapidjson::SizeType i = 0, end = doc.Size(); i < end; ++i) {
        const auto& path = doc[i];

        const auto& color_member = path.FindMember("color");
        if (color_member == path.MemberEnd()) {
            throw std::runtime_error("Unexpected annotation paths json content - no color found");
        }
        const auto& color = color_member->value;
        const auto alpha_member = color.FindMember("a");
        const auto red_member   = color.FindMember("r");
        const auto green_member = color.FindMember("g");
        const auto blue_member  = color.FindMember("b");

        if (alpha_member == color.MemberEnd() || red_member == color.MemberEnd() || green_member == color.MemberEnd() || blue_member == color.MemberEnd()) {
            throw std::runtime_error("Unexpected annotation paths json content - missing color component");
        }

        dlib::rgb_alpha_pixel rgba_label;
        rgba_label.alpha = alpha_member->value.GetInt();
        rgba_label.red   = red_member  ->value.GetInt();
        rgba_label.green = green_member->value.GetInt();
        rgba_label.blue  = blue_member ->value.GetInt();

        const size_t index_label = rgba_label_to_index_label(rgba_label, anno_classes);

        const auto& color_paths_member = path.FindMember("color_paths");
        if (color_paths_member == path.MemberEnd()) {
            throw std::runtime_error("Unexpected annotation paths json content - no color_paths member found");
        }
        const auto& color_paths = color_paths_member->value;
        if (!color_paths.IsArray() || color_paths.Size() != 1) {
            throw std::runtime_error("Unexpected annotation paths json content - color_paths should be an array having a length of exactly 1");
        }

        double min_x =  std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double min_y =  std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();

        const auto& color_path = color_paths[0];

        if (!color_path.IsArray() || color_path.Size() == 0) {
            throw std::runtime_error("Unexpected annotation paths json content - color_paths elements should not be empty");
        }

        for (rapidjson::SizeType j = 0, end = color_path.Size(); j < end; ++j) {
            const auto& point = color_path[j];
            const auto& x_member = point.FindMember("x");
            const auto& y_member = point.FindMember("y");
            if (x_member == point.MemberEnd() || y_member == point.MemberEnd()) {
                throw std::runtime_error("Unexpected annotation paths json content - color_paths points must have x and y coordinates");
            }
            const double x = x_member->value.GetDouble();
            const double y = y_member->value.GetDouble();
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }

        const long left   = static_cast<long>(std::round(min_x));
        const long right  = static_cast<long>(std::round(max_x));
        const long top    = static_cast<long>(std::round(min_y));
        const long bottom = static_cast<long>(std::round(max_y));

        dlib::mmod_rect mmod_rect;
        mmod_rect.rect = dlib::rectangle(left, top, right, bottom);
        mmod_rect.label = anno_classes[index_label].classlabel;
        mmod_rect.ignore = index_label == 0 || anno_classes[index_label].classlabel == "<<ignore>>";

        mmod_rects.push_back(mmod_rect);
    }

    return mmod_rects;
}

std::vector<dlib::mmod_rect> downscale_labels(const std::vector<dlib::mmod_rect>& labels, double downscaling_factor)
{
    auto result = labels;

    if (downscaling_factor != 1.0) {
        for (auto& label : result) {
            const auto scale = [downscaling_factor](const long value) {
                return static_cast<long>(std::round(value / downscaling_factor));
            };
            label.rect.set_left(scale(label.rect.left()));
            label.rect.set_right(scale(label.rect.right()));
            label.rect.set_top(scale(label.rect.top()));
            label.rect.set_bottom(scale(label.rect.bottom()));
        }
    }

    return result;
}

void decode_rgba_label_image(const dlib::matrix<dlib::rgb_alpha_pixel>& rgba_label_image, dlib::matrix<uint16_t>& indexed_label_image, const std::vector<AnnoClass>& anno_classes)
{
    const long nr = rgba_label_image.nr();
    const long nc = rgba_label_image.nc();

    indexed_label_image.set_size(nr, nc);

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            const uint16_t label = rgba_label_to_index_label(rgba_label_image(r, c), anno_classes);
            indexed_label_image(r, c) = label;
        }
    }
}

sample read_sample(const image_filenames& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth, double downscaling_factor)
{
    sample sample;
    sample.image_filenames = image_filenames;

    try {
        dlib::matrix<dlib::rgb_alpha_pixel> rgba_label_image;
        dlib::load_image(sample.input_image, image_filenames.image_filename);
        sample.original_width = sample.input_image.nc();
        sample.original_height = sample.input_image.nr();
        dlib::resize_image(1.0 / downscaling_factor, sample.input_image);

        if (!image_filenames.label_filename.empty()) {
            const std::string json = read_file_as_string(image_filenames.label_filename);
            sample.labels = downscale_labels(parse_labels(json, anno_classes), downscaling_factor);
        }
        else if (require_ground_truth) {
            sample.error = "No ground truth available";
        }

        if (!image_filenames.segmentation_label_filename.empty()) {
            // TODO: optimize by avoiding unnecessary memory re-allocations for the temp matrixes?
            dlib::matrix<dlib::rgb_alpha_pixel> temp1;
            dlib::load_image(temp1, image_filenames.segmentation_label_filename);
            dlib::matrix<uint16_t> temp2;
            decode_rgba_label_image(temp1, temp2, anno_classes);
            sample.segmentation_labels.set_size(sample.input_image.nr(), sample.input_image.nc());
            dlib::resize_image(temp2, sample.segmentation_labels, dlib::interpolate_nearest_neighbor());

            dlib::matrix<uint32_t> temp3;
            dlib::label_connected_blobs(temp2, dlib::zero_pixels_are_background(), dlib::neighbors_8(), dlib::connected_if_equal(), temp3);
            sample.connected_label_components.set_size(sample.input_image.nr(), sample.input_image.nc());
            dlib::resize_image(temp3, sample.connected_label_components, dlib::interpolate_nearest_neighbor());
        }
    }
    catch (std::exception& e) {
        sample.error = e.what();
    }

    return sample;
};

dlib::rectangle get_cropping_rect(const dlib::rectangle& rectangle)
{
    DLIB_ASSERT(!rectangle.is_empty());

    const auto center_point = dlib::center(rectangle);
    const auto max_dim = std::max(rectangle.width(), rectangle.height());
    const auto d = static_cast<long>(std::round(max_dim / 2.0 * 1.5)); // add +50%

    return dlib::rectangle(
        center_point.x() - d,
        center_point.y() - d,
        center_point.x() + d,
        center_point.y() + d
    );
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