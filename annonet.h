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

#ifndef ANNONET_H
#define ANNONET_H

#include <dlib/dnn.h>
#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include "dlib-dnn-pimpl-wrapper-for-segmentation/NetPimpl.h"
#include <unordered_map>
#include "annonet_parse_anno_classes.h"

// ----------------------------------------------------------------------------------------

struct image_filenames
{
    std::string image_filename;
    std::string label_filename;
    std::string segmentation_label_filename;
    std::vector<dlib::mmod_rect> labels;
};

typedef uint8_t input_pixel_type;

struct sample
{
    int original_width = 0;
    int original_height = 0;
    image_filenames image_filenames;
    NetPimpl::input_type input_image;
    std::vector<dlib::mmod_rect> labels;
    dlib::matrix<uint16_t> segmentation_labels;
    dlib::matrix<uint32_t> connected_label_components;
    std::string error;
};

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes);

std::vector<dlib::mmod_rect> parse_labels(const std::string& json, const std::vector<AnnoClass>& anno_classes);

std::vector<dlib::mmod_rect> downscale_labels(const std::vector<dlib::mmod_rect>& labels, double downscaling_factor);

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

sample read_sample(const image_filenames& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth, double downscaling_factor);

dlib::rectangle get_cropping_rect(const dlib::rectangle& rectangle, double max_relative_instance_size);

void set_low_priority();

#endif // ANNONET_H