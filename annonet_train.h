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
#include <numeric>

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
