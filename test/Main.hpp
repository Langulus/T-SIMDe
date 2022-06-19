///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include <Langulus.T-SIMDe.hpp>
#include <cstdint>
#include <cstddef>

using namespace Langulus;

//#define LANGULUS_STD_BENCHMARK

#define CATCH_CONFIG_ENABLE_BENCHMARKING

#define SIGNED_TYPES() ::std::int8_t, ::std::int16_t, ::std::int32_t, ::std::int64_t, float, double
#define UNSIGNED_TYPES() ::std::uint8_t, ::std::uint16_t, ::std::uint32_t, ::std::uint64_t, ::std::byte, char8_t, char16_t, char32_t, wchar_t

#define SPARSE_SIGNED_TYPES() ::std::int8_t*, ::std::int16_t*, ::std::int32_t*, ::std::int64_t*, float*, double*
#define SPARSE_UNSIGNED_TYPES() ::std::uint8_t*, ::std::uint16_t*, ::std::uint32_t*, ::std::uint64_t*, ::std::byte*, char8_t*, char16_t*, char32_t*, wchar_t*

using uint = unsigned int;
template<class T>
using some = std::vector<T>;

template<class T, class A>
void InitOne(T& a, A&& b) noexcept {
	if constexpr (CT::Sparse<T>) {
		using DT = Decay<T>;
		a = new DT {static_cast<DT>(b)};
	}
	else a = static_cast<T>(b);
}
