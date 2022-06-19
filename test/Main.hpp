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

#ifdef LANGULUS_STD_BENCHMARK
	#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif

#define SIGNED_TYPES() ::std::int8_t, ::std::int16_t, ::std::int32_t, ::std::int64_t, float, double
#define UNSIGNED_TYPES() ::std::uint8_t, ::std::uint16_t, ::std::uint32_t, ::std::uint64_t, ::std::byte, char8_t, char16_t, char32_t, wchar_t

#define SPARSE_SIGNED_TYPES() ::std::int8_t*, ::std::int16_t*, ::std::int32_t*, ::std::int64_t*, float*, double*
#define SPARSE_UNSIGNED_TYPES() ::std::uint8_t*, ::std::uint16_t*, ::std::uint32_t*, ::std::uint64_t*, ::std::byte*, char8_t*, char16_t*, char32_t*, wchar_t*

template<class T, class HEAD, class... TAIL>
void InitInner(T& a, HEAD&& head, TAIL&&... tail) noexcept {
	if constexpr (CT::Sparse<T>) {
		using DT = Decay<T>;
		a = new DT {static_cast<DT>(head)};
	}
	else a = static_cast<T>(head);

	if constexpr (0 == sizeof...(TAIL))
		return;
	else
		InitInner(*((&a) + 1), tail...);
}

template<class T, size_t C, class... A>
void Init(T(&a)[C], A&&... arguments) noexcept {
	static_assert(sizeof...(A) == C, "Wrong number of arguments");
	InitInner(a[0], arguments...);
}

template<class T, class A>
void InitOne(T& a, A&& b) noexcept {
	if constexpr (CT::Sparse<T>) {
		using DT = Decay<T>;
		a = new DT {static_cast<DT>(b)};
	}
	else a = static_cast<T>(b);
}

template<class T, size_t C>
void Free(T(&a)[C]) noexcept {
	if constexpr (CT::Sparse<T>) {
		for (auto& i : a)
			delete i;
	}
}

