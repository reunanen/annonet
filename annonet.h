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

struct image_filenames_type
{
    std::string image_filename;
    std::string label_filename;
};

typedef uint8_t input_pixel_type;

struct sample_type
{
    int original_width = 0;
    int original_height = 0;
    image_filenames_type image_filenames;
    NetPimpl::input_type input_image;
    dlib::matrix<uint16_t> label_image;
    std::unordered_map<uint16_t, std::deque<dlib::point>> labeled_points_by_class;
    std::string error;
};

inline uint16_t rgba_label_to_index_label(const dlib::rgb_alpha_pixel& rgba_label, const std::vector<AnnoClass>& anno_classes);

void decode_rgba_label_image(const dlib::matrix<dlib::rgb_alpha_pixel>& rgba_label_image, sample_type& ground_truth_sample, const std::vector<AnnoClass>& anno_classes);

std::vector<image_filenames_type> find_image_files(
    const std::string& anno_data_folder,
    bool require_ground_truth
);

template <typename image_type>
void resize_label_image(image_type& label_image, int target_width, int target_height);

sample_type read_sample(const image_filenames_type& image_filenames, const std::vector<AnnoClass>& anno_classes, bool require_ground_truth, double downscaling_factor);

template <
    typename image_type
>
void outpaint(
    dlib::image_view<image_type>& img,
    dlib::rectangle inside
)
{
    inside = inside.intersect(get_rect(img));
    if (inside.is_empty())
    {
        return;
    }

    for (long r = 0; r < inside.top(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[inside.top()][inside.left()];
        }
        for (long c = inside.left(); c <= inside.right(); ++c) {
            img[r][c] = img[inside.top()][c];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[inside.top()][inside.right()];
        }
    }
    for (long r = inside.top(); r <= inside.bottom(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[r][inside.left()];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[r][inside.right()];
        }
    }
    for (long r = inside.bottom() + 1; r < img.nr(); ++r) {
        for (long c = 0; c < inside.left(); ++c) {
            img[r][c] = img[inside.bottom()][inside.left()];
        }
        for (long c = inside.left(); c <= inside.right(); ++c) {
            img[r][c] = img[inside.bottom()][c];
        }
        for (long c = inside.right() + 1; c < img.nc(); ++c) {
            img[r][c] = img[inside.bottom()][inside.right()];
        }
    }

    // TODO: even blur from outside
}

void set_low_priority();

#endif // ANNONET_H
