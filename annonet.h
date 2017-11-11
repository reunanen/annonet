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

#ifndef ANNONET_H
#define ANNONET_H

#include <dlib/dnn.h>
#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include <unordered_map>

// ----------------------------------------------------------------------------------------

inline bool operator == (const dlib::rgb_alpha_pixel& a, const dlib::rgb_alpha_pixel& b);

// ----------------------------------------------------------------------------------------

struct zero_and_ignored_pixels_are_background
{
    template <typename image_type>
    bool operator() (
        const image_type& label_image,
        const dlib::point& point
    ) const
    {
        const uint16_t label = label_image[point.y()][point.x()];
        return label == 0 || label == dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
    }
};

// ----------------------------------------------------------------------------------------

struct AnnoClass {
    AnnoClass(uint16_t index, const dlib::rgb_alpha_pixel& rgba_label, const std::string& classlabel)
        : index(index), rgba_label(rgba_label), classlabel(classlabel)
    {}

    const uint16_t index = 0;
    const dlib::rgb_alpha_pixel rgba_label;
    const std::string classlabel;
};

namespace {
    dlib::rgb_alpha_pixel rgba_ignore_label(0, 0, 0, 0);
}

std::vector<AnnoClass> parse_anno_classes(const std::string& json);

struct image_filenames
{
    std::string image_filename;
    std::string label_filename;
};

typedef uint8_t input_pixel_type;

struct sample
{
    int original_width = 0;
    int original_height = 0;
    image_filenames image_filenames;
    NetPimpl::input_type input_image;
    dlib::matrix<uint16_t> label_image;
    std::unordered_map<uint16_t, std::deque<dlib::point>> labeled_points_by_class;
    std::string error;
};

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes);

void decode_rgba_label_image(const dlib::matrix<dlib::rgb_alpha_pixel>& rgba_label_image, sample& ground_truth_sample, const std::vector<AnnoClass>& anno_classes);

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

template <typename image_type>
void resize_label_image(image_type& label_image, int target_width, int target_height);

sample read_sample(const image_filenames& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth, double downscaling_factor);

#endif // ANNONET_H