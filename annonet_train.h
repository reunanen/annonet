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

#include "dlib-dnn-pimpl-wrapper/NetPimpl.h"
#include <unordered_map>
#include <numeric>

void set_weights (
    const dlib::matrix<uint16_t>& unweighted_label_image,
    NetPimpl::training_label_type& weighted_label_image,
    double class_weight, // Try 0.0 for equally balanced pixels, and 1.0 for equally balanced classes
    double image_weight  // Try 0.0 for equally balanced pixels, and 1.0 for equally balanced images
)
{
    const long nr = unweighted_label_image.nr();
    const long nc = unweighted_label_image.nc();

    std::unordered_map<uint16_t, size_t> label_counts;

    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            const uint16_t label = unweighted_label_image(r, c);
            if (label != dlib::loss_multiclass_log_per_pixel_::label_to_ignore) {
                ++label_counts[label];
            }
        }
    }

    const size_t total_count = std::accumulate(label_counts.begin(), label_counts.end(), 0,
        [&](size_t total, const std::pair<uint16_t, size_t>& item) { return total + item.second; });

    std::unordered_map<uint16_t, double> label_weights;

    if (total_count > 0) {
        const double average_count = total_count / static_cast<double>(label_counts.size());

        double total_unnormalized_weight = 0.0;
        for (const auto& item : label_counts) {
            const double unnormalized_label_weight = pow(average_count / item.second, class_weight);
            label_weights[item.first] = unnormalized_label_weight;
            total_unnormalized_weight += item.second * unnormalized_label_weight;
        }

        // normalize label weights
        const double target_total_weight = total_count * pow(nr * nc / static_cast<double>(total_count), image_weight);
        for (auto& item : label_weights) {
            item.second *= target_total_weight / total_unnormalized_weight;
        }
    }

    weighted_label_image.set_size(nr, nc);

    for (int r = 0; r < nr; ++r) {
        for (int c = 0; c < nc; ++c) {
            const uint16_t label = unweighted_label_image(r, c);
            const double weight = label == dlib::loss_multiclass_log_per_pixel_::label_to_ignore ? 0.0 : label_weights[label];
            weighted_label_image(r, c) = dlib::loss_multiclass_log_per_pixel_weighted_::weighted_label(label, weight);
        }
    }
}

dlib::rectangle random_rect_containing_point(
    dlib::rand& rnd,
    const dlib::point& point,
    long result_width,
    long result_height,
    const dlib::rectangle& limits
)
{
    DLIB_ASSERT(limits.contains(point));
    DLIB_ASSERT(result_width <= limits.width());
    DLIB_ASSERT(result_height <= limits.height());
    const long min_center_x = std::max(limits.left() + result_width / 2, point.x() - (result_width - 1) / 2);
    const long max_center_x = std::min(limits.right() - (result_width - 1) / 2, point.x() + result_width / 2);
    const long min_center_y = std::max(limits.top() + result_height / 2, point.y() - (result_height - 1) / 2);
    const long max_center_y = std::min(limits.bottom() - (result_height - 1) / 2, point.y() + result_height / 2);
    DLIB_ASSERT(max_center_x >= min_center_x);
    DLIB_ASSERT(max_center_y >= min_center_y);
    const long center_x = min_center_x + rnd.get_random_32bit_number() % (max_center_x - min_center_x + 1);
    const long center_y = min_center_y + rnd.get_random_32bit_number() % (max_center_y - min_center_y + 1);
    const auto rect = dlib::centered_rect(dlib::point(center_x, center_y), result_width, result_height);
    DLIB_ASSERT(rect.width() == result_width);
    DLIB_ASSERT(rect.height() == result_height);
    DLIB_ASSERT(limits.contains(rect));
    DLIB_ASSERT(rect.contains(point));
    return rect;
};
