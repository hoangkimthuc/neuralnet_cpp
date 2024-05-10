#include "include/ops.h"
#include <gtest/gtest.h>

TEST(Linear_forward, Initialization) {
    // Set the weights, bias
    Linear linear(2, 2);
    EXPECT_EQ(linear.weights.data.size(), 4);
    
}
