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
#include <limits>
#include <stdexcept>

namespace pipeline {

// ============================================================================
// Poisson utilities
// ============================================================================
/// Compute the Poisson CDF: P(N <= k | lambda)
inline double poisson_cdf(int k, double lambda) {
  if (k < 0) return 0.0;
  double term = std::exp(-lambda);  // i = 0 term
  double sum  = term;
  for (int i = 1; i <= k; ++i) {
    term *= lambda / static_cast<double>(i);
    sum  += term;
  }
  return sum;
}

/// Survival function P(N >= T | lambda)
inline double poisson_sf_ge(int T, double lambda) {
  if (T <= 0) return 1.0;
  return 1.0 - poisson_cdf(T - 1, lambda);
}

// ============================================================================
// PoissonDetector
// ============================================================================
class PoissonDetector {
public:
  /// Constructs a detector with the following parameters:
  /// @param lambda_occ The mean photons when site is occupied
  /// @param lambda_empty The mean photons when site is empty
  /// @param fp_target The max allowed false-positive rate P(N >= T | empty)
  /// @param max_k The maximum candidate threshold to scan
  /// @return A new PoissonDetector object
  PoissonDetector(double lambda_occ, double lambda_empty, double fp_target, 
                  int max_k = 50)
  : lambda_occ_{lambda_occ}, lambda_empty_{lambda_empty}, fp_target_{fp_target}, 
    max_k_{max_k}, threshold_{compute_threshold()} {}

  /// Integer photon-count threshold T.
  [[nodiscard]] int threshold() const noexcept {
    return threshold_;
  }

  /// Classify an ROI sum: true = occupied, false = empty.
  [[nodiscard]] bool classify(double roi_sum) const noexcept {
    return roi_sum >= static_cast<double>(threshold_);
  }

  /// P(N >= T | empty) with current parameters.
  [[nodiscard]] double false_positive_rate() const {
    return poisson_sf_ge(threshold_, lambda_empty_);
  }

  /// P(N < T | occupied) with current parameters.
  [[nodiscard]] double false_negative_rate() const {
    return poisson_cdf(threshold_ - 1, lambda_occ_);
  }

private:
  double lambda_occ_;
  double lambda_empty_;
  double fp_target_;
  int    max_k_;
  int    threshold_;

  /// Find threshold T in [0, max_k_] satisfying the FP constraint and
  /// minimizing FP + FN. Throws if no T satisfies fp_target_.
  [[nodiscard]] int compute_threshold() const {
    int    best_T    = -1;
    double best_loss = std::numeric_limits<double>::infinity();

    for (int T = 0; T <= max_k_; ++T) {
      const double p_fp = poisson_sf_ge(T, lambda_empty_);
      if (p_fp > fp_target_) continue;

      const double p_fn  = poisson_cdf(T - 1, lambda_occ_);
      const double loss  = p_fp + p_fn;
      if (loss < best_loss) { best_loss = loss; best_T    = T;}
    }

    if (best_T < 0) {
      throw std::runtime_error(
        "PoissonDetector: no threshold satisfies the false-positive "
        "constraint. Increase max_k or fp_target."
      );
    }

    return best_T;
  }
};

} // namespace pipeline
