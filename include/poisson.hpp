// ============================================================================
// poisson_detector.hpp -- Poisson Detector for ROI occupancy detection
//
// This class implements a Poisson detector for ROI occupancy detection.
// It uses the Poisson distribution to model the number of photons in a ROI
// and finds an integer threshold T that satisfies a false-positive constraint
// while minimizing the false-negative and false-positive rates.
// ============================================================================
#pragma once
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pipeline {

// ============================================================================
// `ROI` class
// Rectangle ROI in image pixel coords (row-major H×W, uint16 pixels)
// ============================================================================
struct ROI {
  std::size_t x{0}, y{0}, w{0}, h{0};
  uint32_t threshold{0};   // classify occupied if sum >= threshold
};

// ============================================================================
// Poisson utilities
// ============================================================================
/// Compute the Poisson CDF: P(N <= k | lambda)
template<typename FloatType = double>
inline FloatType poisson_cdf(int k, FloatType lambda) {
  if (k < 0) return FloatType(0.0);
  FloatType term = std::exp(-lambda);  // i = 0 term
  FloatType sum  = term;
  for (int i = 1; i <= k; ++i) {
    term *= lambda / static_cast<FloatType>(i);
    sum  += term;
  }
  return sum;
}

/// Survival function P(N >= T | lambda)
template<typename FloatType = double>
inline FloatType poisson_sf_ge(int T, FloatType lambda) {
  if (T <= 0) return FloatType(1.0);
  return FloatType(1.0) - poisson_cdf<FloatType>(T - 1, lambda);
}

// ============================================================================
// PoissonDetector
// ============================================================================
/// @tparam FloatType Floating point type for calculations (default: double)
/// @tparam IntType Integer type for threshold (default: int)
template<typename FloatType = double, typename IntType = int>
class PoissonDetector {
public:
  /// Constructs a detector with the following parameters:
  /// @param lambda_occ   The mean photons when site is occupied
  /// @param lambda_empty The mean photons when site is empty
  /// @param fp_target    The max allowed false-positive rate P(N >= T | empty)
  /// @param max_k        The maximum candidate threshold to scan 
  PoissonDetector(FloatType lambda_occ, FloatType lambda_empty, FloatType fp_target, 
                  IntType max_k = 50)
  : lambda_occ_{lambda_occ}, lambda_empty_{lambda_empty}, fp_target_{fp_target}, 
    max_k_{max_k}, threshold_{compute_threshold()} {}

  /// Integer photon-count threshold T.
  [[nodiscard]] IntType threshold() const noexcept {
    return threshold_;
  }

  /// Classify an ROI sum: true = occupied, false = empty.
  template<typename SumType>
  [[nodiscard]] bool classify(SumType roi_sum) const noexcept {
    return static_cast<FloatType>(roi_sum) >= static_cast<FloatType>(threshold_);
  }

  /// P(N >= T | empty) with current parameters.
  [[nodiscard]] FloatType false_positive_rate() const {
    return poisson_sf_ge<FloatType>(threshold_, lambda_empty_);
  }

  /// P(N < T | occupied) with current parameters.
  [[nodiscard]] FloatType false_negative_rate() const {
    return poisson_cdf<FloatType>(threshold_ - 1, lambda_occ_);
  }

private:
  FloatType lambda_occ_;
  FloatType lambda_empty_;
  FloatType fp_target_;
  IntType   max_k_;
  IntType   threshold_;

  /// Find threshold T in [0, max_k_] satisfying the FP constraint and
  /// minimizing FP + FN. Throws if no T satisfies fp_target_.
  [[nodiscard]] IntType compute_threshold() const {
    IntType    best_T    = -1;
    FloatType best_loss = std::numeric_limits<FloatType>::infinity();

    for (IntType T = 0; T <= max_k_; ++T) {
      const FloatType p_fp = poisson_sf_ge<FloatType>(T, lambda_empty_);
      if (p_fp > fp_target_) continue;

      const FloatType p_fn  = poisson_cdf<FloatType>(T - 1, lambda_occ_);
      const FloatType loss  = p_fp + p_fn;
      if (loss < best_loss) { best_loss = loss; best_T    = T;}
    }

    if (best_T < 0) {
      throw std::runtime_error(
        "POIS: no threshold satisfies the false-positive "
        "constraint. Increase max_k or fp_target."
      );
    }

    return best_T;
  }
};

// ============================================================================
// ROI building utilities
// Build a grid of ROIs and assign Poisson thresholds
// ============================================================================
/// @param row_start      The start row of the ROI grid
/// @param col_start      The start column of the ROI grid
/// @param row_end        The end row of the ROI grid
/// @param col_end        The end column of the ROI grid
/// @param roi_w          The width of each ROI
/// @param roi_h          The height of each ROI
/// @param lambda_occ     The mean photons when site is occupied
/// @param lambda_empty   The mean photons when site is empty
/// @param fp_target      The maximum allowed false-positive rate
/// @param max_k          The maximum candidate threshold to scan
/// @return               A vector of ROIs
inline std::vector<ROI> build_rois_with_poisson(
    std::size_t row_start, std::size_t col_start,
    std::size_t row_end, std::size_t col_end,
    std::size_t roi_w, std::size_t roi_h,
    double lambda_occ, double lambda_empty,
    double fp_target, int max_k) {
  
  std::vector<ROI> rois;
  for (std::size_t y = row_start; y + roi_h <= row_end; y += roi_h) {
    for (std::size_t x = col_start; x + roi_w <= col_end; x += roi_w) {
      ROI r{x, y, roi_w, roi_h, 0};
      rois.emplace_back(r);
    }
  }
  std::printf("POIS: Built %zu ROIs\n", rois.size());

  PoissonDetector<> det(lambda_occ, lambda_empty, fp_target, max_k);
  const int T = det.threshold();

  std::printf("POIS: PoissonDetector threshold T = %d "
              "(FP=%.3e, FN=%.3e)\n",
              T,
              det.false_positive_rate(),
              det.false_negative_rate());

  for (auto& r : rois) {
    r.threshold = static_cast<std::uint32_t>(T);
  }

  std::printf("POIS: Configured %zu ROIs of size %zux%zu\n",
              rois.size(), roi_w, roi_h);
  return rois;
}

} // namespace pipeline
