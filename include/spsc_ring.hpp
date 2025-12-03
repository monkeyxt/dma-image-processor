// ============================================================================
// spsc_ring.hpp -- Single-Producer Single-Consumer Ring Buffer
// ============================================================================
#pragma once
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace spsc {

#if defined(__cpp_lib_hardware_interference_size)
constexpr std::size_t CL = std::hardware_destructive_interference_size;
#else
constexpr std::size_t CL = 64;
#endif

// Fixed-capacity SPSC ring (capacity must be a power of two).
// T should be cheap to move or trivially copyable.
template <class T, std::size_t CapacityPow2>
class Ring {
  static_assert((CapacityPow2 & (CapacityPow2 - 1)) == 0,
                "Capacity must be a power of two");
  static_assert(CapacityPow2 >= 2, "Capacity too small");

public:
  Ring() = default;
  Ring(const Ring&) = delete;
  Ring& operator=(const Ring&) = delete;

  /// Emplace in-place into the ring.
  /// @param args The arguments to construct the object with.
  /// @return True if the object was emplaced, false if the ring is full.
  template <class... Args>
  bool emplace(Args&&... args) noexcept(
    std::is_nothrow_constructible_v<T, Args...>) {
    auto h = head_.load(std::memory_order_relaxed);
    auto next = (h + 1) & MASK;
    if (next == tail_.load(std::memory_order_acquire)) return false; // full
    new (&buf_[h]) T(std::forward<Args>(args)...);
    head_.store(next, std::memory_order_release);
    return true;
  }

  /// Push by copy/move.
  /// @param v The value to push.
  /// @return True if the value was pushed, false if the ring is full.
  bool push(const T& v) noexcept(std::is_nothrow_copy_constructible_v<T>) { 
    return emplace(v); 
  }

  /// Push by move.
  /// @param v The value to push.
  /// @return True if the value was pushed, false if the ring is full.
  bool push(T&& v) noexcept(std::is_nothrow_move_constructible_v<T>) { 
    return emplace(std::move(v)); 
  }

  /// Pop into out param.
  /// @param out The value to pop into.
  /// @return True if the value was popped, false if the ring is empty.
  bool pop(T& out) noexcept {
    auto t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return false; // empty
    T* p = &buf_[t];
    out = std::move(*p);
    p->~T();
    tail_.store((t + 1) & MASK, std::memory_order_release);
    return true;
  }

  /// Optional peek (single-consumer safe); caller must call drop_one() after.
  /// @return A pointer to the value at the tail of the ring, or nullptr.
  T* peek() noexcept {
    auto t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return nullptr;
    return &buf_[t];
  }

  /// Drop the value at the tail of the ring.
  /// @return True if the value was dropped, false if the ring is empty.
  bool drop_one() noexcept {
    auto t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return false;
    buf_[t].~T();
    tail_.store((t + 1) & MASK, std::memory_order_release);
    return true;
  }

  /// Non-blocking bulk ops (handy for bursts).
  /// @param src The source array to push.
  /// @param n The number of values to push.
  /// @return The number of values pushed.
  std::size_t push_bulk(const T* src, std::size_t n) noexcept {
    std::size_t pushed = 0;
    while (pushed < n && push(src[pushed])) ++pushed;
    return pushed;
  }

  /// Pop a bulk of values from the ring.
  /// @param dst The destination array to pop into.
  /// @param n The number of values to pop.
  /// @return The number of values popped.
  std::size_t pop_bulk(T* dst, std::size_t n) noexcept {
    std::size_t popped = 0;
    while (popped < n && pop(dst[popped])) ++popped;
    return popped;
  }

  /// Capacity is N-1 (one slot left empty to disambiguate full vs. empty).
  /// @return The capacity of the ring.
  static constexpr std::size_t capacity() noexcept { return CapacityPow2 - 1; }

  /// Check if the ring is empty.
  /// @return True if the ring is empty, false otherwise.
  bool empty() const noexcept {
    return tail_.load(std::memory_order_acquire) == 
                      head_.load(std::memory_order_acquire);
  }

  /// Check if the ring is full.
  /// @return True if the ring is full, false otherwise.
  bool full() const noexcept {
    auto h = head_.load(std::memory_order_acquire);
    return ((h + 1) & MASK) == tail_.load(std::memory_order_acquire);
  }

private:
  static constexpr std::size_t MASK = CapacityPow2 - 1;

  alignas(CL) std::atomic<std::size_t> head_{0}; // producer writes, consumer reads
  alignas(CL) std::atomic<std::size_t> tail_{0}; // consumer writes, producer reads
  alignas(CL) std::byte storage_[sizeof(T) * CapacityPow2]; // raw storage

  /// placement array access
  T& buf_at(std::size_t i) noexcept { 
    return *std::launder(reinterpret_cast<T*>(storage_) + i); 
  }
  const T& buf_at(std::size_t i) const noexcept { 
    return *std::launder(reinterpret_cast<const T*>(storage_) + i); 
  }

  /// convenience to index like an array
  T& operator[](std::size_t i) noexcept { return buf_at(i); }
  const T& operator[](std::size_t i) const noexcept { return buf_at(i); }

  T buf_[CapacityPow2];
};

} // namespace spsc
