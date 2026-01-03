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
#include <unordered_map>
#include "annonet_parse_anno_classes.h"

// ----------------------------------------------------------------------------------------

struct image_filenames_type
{
    std::string image_filename;
    std::string label_filename;
    std::vector<dlib::mmod_rect> labels;
};

typedef uint8_t input_pixel_type;

struct sample_type
{
    int original_width = 0;
    int original_height = 0;
    image_filenames_type image_filenames;
    NetPimpl::input_type input_image;
    std::vector<dlib::mmod_rect> labels;
    std::string error;
};

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes);

std::vector<dlib::mmod_rect> parse_labels(const std::string& json, const std::vector<AnnoClass>& anno_classes);

std::vector<image_filenames_type> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

sample_type read_sample(const image_filenames_type& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth, double downscaling_factor);

void set_low_priority();

#endif // ANNONET_H