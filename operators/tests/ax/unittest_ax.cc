// Copyright Axelera AI, 2023

#include <gtest/gtest.h>
#include <glib.h>

/**
 * @brief Main function for unit test.
 */
int
main(int argc, char **argv)
{
  int ret = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    g_warning("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  try {
    ret = RUN_ALL_TESTS();
  } catch (...) {
    g_warning("catch `testing::internal::GoogleTestFailureException`");
  }

  return ret;
}
