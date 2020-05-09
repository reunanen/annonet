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
#include "annonet_parse_anno_classes.h"

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

struct image_filenames
{
    std::string image_filename;
    std::string classlabel;
};

typedef uint8_t input_pixel_type;

struct sample
{
    int original_width = 0;
    int original_height = 0;
    image_filenames image_filenames;
    NetPimpl::input_type input_image;
    unsigned long classlabel = std::numeric_limits<unsigned long>::max();
    std::string error;
};

std::vector<image_filenames> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

sample read_sample(const image_filenames& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth);

void convert_for_processing(
    const NetPimpl::input_type& full_input_image,
    NetPimpl::input_type& converted,
    int dim
);

void set_low_priority();

#endif // ANNONET_H