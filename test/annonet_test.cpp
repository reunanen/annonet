#include "../annonet_train.h"
#include "picotest/picotest.h"

namespace {

    class TrainTest : public ::testing::Test {
    protected:
        // You can remove any or all of the following functions if its body
        // is empty.

        TrainTest() {
            unweighted_label_image.set_size(1, 5);
            unweighted_label_image(0, 0) = 0;
            unweighted_label_image(0, 1) = dlib::loss_multiclass_log_per_pixel_::label_to_ignore;
            unweighted_label_image(0, 2) = 1;
            unweighted_label_image(0, 3) = 0;
            unweighted_label_image(0, 4) = 0;
        }

        virtual ~TrainTest() {
            // You can do clean-up work that doesn't throw exceptions here.
        }

        // If the constructor and destructor are not enough for setting up
        // and cleaning up each test, you can define the following methods:

        virtual void SetUp() {
            // Code here will be called immediately after the constructor (right
            // before each test).
        }

        virtual void TearDown() {
            // Code here will be called immediately after each test (right
            // before the destructor).
        }

        // Objects declared here can be used by all tests of TrainTest.
        dlib::matrix<uint16_t> unweighted_label_image;
    };

    static double GetTotalWeight(const NetPimpl::training_label_type& weighted_label_image)
    {
        const long nr = weighted_label_image.nr();
        const long nc = weighted_label_image.nc();
        double total_weight = 0;
        for (long r = 0; r < nr; ++r) {
            for (long c = 0; c < nc; ++c) {
                total_weight += weighted_label_image(r, c).weight;
            }
        }
        return total_weight;
    }

    TEST_F(TrainTest, WeighsPixelsEquivalent) {
        NetPimpl::training_label_type weighted_label_image;

        set_weights(unweighted_label_image, weighted_label_image, 0.0, 0.0);

        EXPECT_EQ(weighted_label_image.nr(), unweighted_label_image.nr());
        EXPECT_EQ(weighted_label_image.nc(), unweighted_label_image.nc());

        EXPECT_EQ(weighted_label_image(0, 0).weight, 1.0);
        EXPECT_EQ(weighted_label_image(0, 1).weight, 0.0);
        EXPECT_EQ(weighted_label_image(0, 2).weight, 1.0);
        EXPECT_EQ(weighted_label_image(0, 3).weight, 1.0);
        EXPECT_EQ(weighted_label_image(0, 4).weight, 1.0);

        EXPECT_EQ(GetTotalWeight(weighted_label_image), 4.0);
    }

    TEST_F(TrainTest, WeighsClassesEquivalent) {
        NetPimpl::training_label_type weighted_label_image;

        set_weights(unweighted_label_image, weighted_label_image, 1.0, 0.0);

        EXPECT_EQ(weighted_label_image.nr(), unweighted_label_image.nr());
        EXPECT_EQ(weighted_label_image.nc(), unweighted_label_image.nc());

        EXPECT_NEAR(weighted_label_image(0, 0).weight, 0.666667, 1e-6);
        EXPECT_EQ(weighted_label_image(0, 1).weight, 0.0);
        EXPECT_EQ(weighted_label_image(0, 2).weight, 2.0);
        EXPECT_NEAR(weighted_label_image(0, 3).weight, 0.666667, 1e-6);
        EXPECT_NEAR(weighted_label_image(0, 4).weight, 0.666667, 1e-6);

        EXPECT_NEAR(GetTotalWeight(weighted_label_image), 4.0, 1e-6);
    }

    TEST_F(TrainTest, WeighsEvenInBetween) {
        NetPimpl::training_label_type weighted_label_image;

        set_weights(unweighted_label_image, weighted_label_image, 0.5, 0.0);

        EXPECT_EQ(weighted_label_image.nr(), unweighted_label_image.nr());
        EXPECT_EQ(weighted_label_image.nc(), unweighted_label_image.nc());

        EXPECT_NEAR(weighted_label_image(0, 0).weight, 0.845299, 1e-6);
        EXPECT_EQ(weighted_label_image(0, 1).weight, 0.0);
        EXPECT_NEAR(weighted_label_image(0, 2).weight, 0.845299 * sqrt(3), 1e-6);
        EXPECT_NEAR(weighted_label_image(0, 3).weight, 0.845299, 1e-6);
        EXPECT_NEAR(weighted_label_image(0, 4).weight, 0.845299, 1e-6);

        EXPECT_NEAR(GetTotalWeight(weighted_label_image), 4.0, 1e-6);
    }

    TEST_F(TrainTest, WeighsImagesEquivalent) {
        NetPimpl::training_label_type weighted_label_image;

        set_weights(unweighted_label_image, weighted_label_image, 0.0, 1.0);

        EXPECT_EQ(weighted_label_image.nr(), unweighted_label_image.nr());
        EXPECT_EQ(weighted_label_image.nc(), unweighted_label_image.nc());

        EXPECT_EQ(weighted_label_image(0, 0).weight, 1.25);
        EXPECT_EQ(weighted_label_image(0, 1).weight, 0.0);
        EXPECT_EQ(weighted_label_image(0, 2).weight, 1.25);
        EXPECT_EQ(weighted_label_image(0, 3).weight, 1.25);
        EXPECT_EQ(weighted_label_image(0, 4).weight, 1.25);

        EXPECT_EQ(GetTotalWeight(weighted_label_image), 5.0);
    }

    TEST_F(TrainTest, GeneratesRandomRectContainingPoint) {
        dlib::rand rnd;
        dlib::point point(50, 50);
        const long width = 10, height = 10;
        dlib::rectangle rect = random_rect_containing_point(rnd, point, width, height);
        EXPECT_EQ(rect.width(), width);
        EXPECT_EQ(rect.height(), height);
        EXPECT_TRUE(rect.contains(point));
    }

}  // namespace

int main(int argc, char **argv) {
    RUN_ALL_TESTS();
}