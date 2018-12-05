#include "../annonet_train.h"
#include "picotest/picotest.h"

namespace {

    class TrainTest : public ::testing::Test {
    protected:
        // You can remove any or all of the following functions if its body
        // is empty.

        TrainTest() {
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
    };

    TEST_F(TrainTest, GeneratesRandomRectContainingPoint) {
        dlib::rand rnd;
        dlib::point point(50, 50);
        dlib::rectangle limits(20, 20, 80, 80);
        const long width = 10, height = 10;
        dlib::rectangle rect = random_rect_containing_point(rnd, point, width, height, limits);
        EXPECT_EQ(rect.width(), width);
        EXPECT_EQ(rect.height(), height);
        EXPECT_TRUE(limits.contains(rect));
        EXPECT_TRUE(rect.contains(point));
    }

}  // namespace

int main(int argc, char **argv) {
    RUN_ALL_TESTS();
}