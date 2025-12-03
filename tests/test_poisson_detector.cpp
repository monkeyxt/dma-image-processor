// ============================================================================
// test_poisson_detector.cpp -- Test PoissonDetector for ROI classifier
// ============================================================================
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "poisson_detector.hpp"  // adjust path as needed

#define EXPECT_TRUE(x) do{ \
  if(!(x)){ \
    std::fprintf(stderr,"EXPECT_TRUE failed: %s @ %s:%d\n",#x,__FILE__,__LINE__); \
    std::abort(); \
  } \
}while(0)

#define EXPECT_EQ(a,b) do{ \
  auto _va=(a); auto _vb=(b); \
  if(!((_va)==(_vb))){ \
    std::fprintf(stderr,"EXPECT_EQ failed: %s=%lld %s=%lld @ %s:%d\n", \
                 #a,(long long)_va,#b,(long long)_vb,__FILE__,__LINE__); \
    std::abort(); \
  } \
}while(0)

using pipeline::PoissonDetector;

// ============================================================================
// Test 1: basic threshold and FP/FN sanity
// ============================================================================
void test_basic_threshold() {
  // Typical parameters for the atom-imaging problem:
  //   lambda_occ   = 10.0
  //   lambda_empty = 0.2
  //   fp_target    = 1e-3
  //
  // With these, the detector should choose T = 4 (see earlier math checks).
  PoissonDetector det(10.0, 0.2, 1e-3, /*max_k=*/50);

  int T = det.threshold();
  EXPECT_EQ(T, 4);

  // Sanity checks on FP/FN: we don't check exact values here (would need float macros),
  // just that we respect the FP target and FN is non-trivial.
  double p_fp = det.false_positive_rate();
  double p_fn = det.false_negative_rate();

  EXPECT_TRUE(p_fp <= 1e-3 + 1e-6);   // within a tiny numerical cushion
  EXPECT_TRUE(p_fn > 0.0);            // should not be zero
  EXPECT_TRUE(p_fn < 0.1);            // for these params we expect a low FN rate

  std::puts("basic threshold selection: OK");
}

// ============================================================================
// Test 2: classification behavior around the threshold
// ============================================================================
void test_classification_edges() {
  PoissonDetector det(10.0, 0.2, 1e-3, /*max_k=*/50);
  int T = det.threshold();
  EXPECT_EQ(T, 4);

  // Below threshold → empty
  EXPECT_TRUE(det.classify(T - 1.0) == false);
  EXPECT_TRUE(det.classify(0.0)     == false);
  EXPECT_TRUE(det.classify(1.5)     == false);

  // At / above threshold → occupied
  EXPECT_TRUE(det.classify(static_cast<double>(T))       == true);
  EXPECT_TRUE(det.classify(static_cast<double>(T) + 0.01)== true);
  EXPECT_TRUE(det.classify(static_cast<double>(T) + 10.0)== true);

  std::puts("classification at threshold edges: OK");
}

// ============================================================================
// Test 3: impossible parameter combo should throw
// ============================================================================
void test_impossible_parameters() {
  bool threw = false;
  try {
    // Very strict fp_target + tiny max_k: no threshold can satisfy P_fp <= fp_target.
    PoissonDetector det(10.0, 1.0, 1e-12, /*max_k=*/2);
    (void)det;
  } catch (const std::runtime_error&) {
    threw = true;
  } catch (...) {
    threw = true;
  }

  EXPECT_TRUE(threw);
  std::puts("impossible parameter handling (throws): OK");
}

// ============================================================================
// Main
// ============================================================================
int main() {
  std::puts("Running PoissonDetector tests...");
  test_basic_threshold();
  test_classification_edges();
  test_impossible_parameters();
  std::puts("All PoissonDetector tests PASSED.");
  return 0;
}
