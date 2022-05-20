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

#define SIGNED_TYPES int8_t, int16_t, int32_t, int64_t, float, double
#define UNSIGNED_TYPES uint8_t, uint16_t, uint32_t, uint64_t, ::std::byte, char8_t, char16_t, char32_t, wchar_t