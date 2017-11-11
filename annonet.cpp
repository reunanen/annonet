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
#include <rapidjson/document.h>

// ----------------------------------------------------------------------------------------

inline bool operator == (const dlib::rgb_alpha_pixel& a, const dlib::rgb_alpha_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue && a.alpha == b.alpha;
}

// ----------------------------------------------------------------------------------------

std::vector<AnnoClass> parse_anno_classes(const std::string& json)
{
    if (json.empty()) {
        // Use the default anno classes
        return std::vector<AnnoClass>{
            AnnoClass(0, dlib::rgb_alpha_pixel(0, 255, 0, 64), "clean"),
            AnnoClass(1, dlib::rgb_alpha_pixel(255, 0, 0, 128), "defect"),
        };
    }

    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        throw std::runtime_error("Error parsing json\n" + json);
    }

    if (!doc.IsObject()) {
        throw std::runtime_error("Unexpected anno classes json content - the document should be an object");
    }

    const auto anno_classes_member = doc.FindMember("anno_classes");

    if (anno_classes_member == doc.MemberEnd() || !anno_classes_member->value.IsArray()) {
        throw std::runtime_error("Unexpected anno classes json content - there should be an anno_classes array");
    }

    std::vector<AnnoClass> anno_classes;

    for (rapidjson::SizeType i = 0, end = anno_classes_member->value.Size(); i < end; ++i) {
        const auto& anno_class = anno_classes_member->value[i];
        const auto name_member = anno_class.FindMember("name");
        const auto color_member = anno_class.FindMember("color");
        if (name_member == anno_class.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - no name found");
        }
        if (color_member == anno_class.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - no color found");
        }
        const auto& color = color_member->value;
        const auto red_member = color.FindMember("red");
        const auto green_member = color.FindMember("green");
        const auto blue_member = color.FindMember("blue");
        const auto alpha_member = color.FindMember("alpha");
        if (red_member == color.MemberEnd() || green_member == color.MemberEnd() || blue_member == color.MemberEnd() || alpha_member == color.MemberEnd()) {
            throw std::runtime_error("Unexpected anno classes json content - color should have all components (red, green, blue, alpha)");
        }
        dlib::rgb_alpha_pixel rgba_value(
            red_member->value.GetInt(),
            green_member->value.GetInt(),
            blue_member->value.GetInt(),
            alpha_member->value.GetInt()
        );

        if (rgba_value == rgba_ignore_label) {
            throw std::runtime_error("Unexpected anno classes json content - rgba (0, 0, 0, 0) is reserved for pixels to be ignored");
        }

        anno_classes.push_back(AnnoClass(i, rgba_value, name_member->value.GetString()));
    }

    return anno_classes;
}

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes)
{
    if (rgba_label == rgba_ignore_label) {
        return dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    }
    for (const AnnoClass& anno_class : anno_classes) {
        if (anno_class.rgba_label == rgba_label) {
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

void decode_rgba_label_image(const dlib::matrix<dlib::rgb_alpha_pixel>& rgba_label_image, sample& ground_truth_sample, const std::vector<AnnoClass>& anno_classes)
{
    const long nr = rgba_label_image.nr();
    const long nc = rgba_label_image.nc();

    ground_truth_sample.label_image.set_size(nr, nc);
    ground_truth_sample.labeled_points_by_class.clear();

    for (long r = 0; r < nr; ++r) {
        for (long c = 0; c < nc; ++c) {
            const uint16_t label = rgba_label_to_index_label(rgba_label_image(r, c), anno_classes);
            if (label != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                ground_truth_sample.labeled_points_by_class[label].push_back(dlib::point(c, r));
            }
            ground_truth_sample.label_image(r, c) = label;
        }
    }
}

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
)
{
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

    std::vector<image_filenames> results;

    const auto file_exists = [](const std::string& filename) {
        std::ifstream label_file(filename, std::ios::binary);
        return !!label_file;
    };

    for (const dlib::file& name : files) {
        image_filenames image_filenames;
        image_filenames.image_filename = name;

        const std::string label_filename = name.full_name() + "_mask.png";
        const bool label_file_exists = file_exists(label_filename);

        if (label_file_exists) {
            image_filenames.label_filename = label_filename;
        }

        if (label_file_exists || !require_ground_truth) {
            results.push_back(image_filenames);
            std::cout << "Added file " << image_filenames.image_filename << std::endl;
        }
        else if (require_ground_truth) {
            std::cout << "Warning: unable to open " << label_filename << std::endl;
        }
    }

    return results;
}

template <typename image_type>
void resize_label_image(image_type& label_image, int target_width, int target_height)
{
    image_type temp;
    dlib::set_image_size(temp, target_height, target_width);
    dlib::resize_image(label_image, temp, dlib::interpolate_nearest_neighbor());
    std::swap(label_image, temp);
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
            dlib::load_image(rgba_label_image, image_filenames.label_filename);

            if (rgba_label_image.nr() != sample.original_height || rgba_label_image.nc() != sample.original_width) {
                sample.error = "Label image size mismatch";
            }
            else {
                resize_label_image(rgba_label_image, sample.input_image.nc(), sample.input_image.nr());
                assert(sample.input_image.nr() == rgba_label_image.nr() || sample.input_image.nc() == rgba_label_image.nc());
                decode_rgba_label_image(rgba_label_image, sample, anno_classes);
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
