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