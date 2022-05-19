///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Intrinsics.hpp"

#if LANGULUS_COMPILER(GCC)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace Langulus::SIMD
{

	/// Fill a register with a single value												
	///	@tparam REGISTER - tyhe of register to fill									
	///	@tparam T - type of data to use for filling									
	///	@param value - the value to use for filling									
	///	@return the filled register														
	template<TSIMD REGISTER, CT::Number T>
	NOD() decltype(auto) Fill(const T& valueOrig) noexcept {
		auto& value = pcVal(valueOrig);
		if constexpr (Same<REGISTER, T>)
			return value;
		else if constexpr (Same<REGISTER, simde__m128i>) {
			if constexpr (Integer8<T>)
				return simde_mm_set1_epi8(value);
			else if constexpr (Integer16<T>)
				return simde_mm_set1_epi16(value);
			else if constexpr (Integer32<T>)
				return simde_mm_set1_epi32(value);
			else if constexpr (Integer64<T>)
				return simde_mm_set1_epi64x(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m128i");
		}
		else if constexpr (Same<REGISTER, simde__m128>) {
			if constexpr (Same<T, float>)
				return simde_mm_broadcast_ss(&value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m128");
		}
		else if constexpr (Same<REGISTER, simde__m128d>) {
			if constexpr (Same<T, double>)
				return simde_mm_set1_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m128d");
		}
		else if constexpr (Same<REGISTER, simde__m256i>) {
			if constexpr (Integer8<T>)
				return simde_mm256_set1_epi8(value);
			else if constexpr (Integer16<T>)
				return simde_mm256_set1_epi16(value);
			else if constexpr (Integer32<T>)
				return simde_mm256_set1_epi32(value);
			else if constexpr (Integer64<T>)
				return simde_mm256_set1_epi64x(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m256i");
		}
		else if constexpr (Same<REGISTER, simde__m256>) {
			if constexpr (Same<T, float>)
				return simde_mm256_broadcast_ss(&value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill __m256");
		}
		else if constexpr (Same<REGISTER, simde__m256d>) {
			if constexpr (Same<T, double>)
				return simde_mm256_broadcast_sd(&value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m256d");
		}
		else if constexpr (Same<REGISTER, simde__m512i>) {
			if constexpr (Integer8<T>)
				return simde_mm512_set1_epi8(value);
			else if constexpr (Integer16<T>)
				return simde_mm512_set1_epi16(value);
			else if constexpr (Integer32<T>)
				return simde_mm512_set1_epi32(value);
			else if constexpr (Integer64<T>)
				return simde_mm512_set1_epi64(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m512i");
		}
		else if constexpr (Same<REGISTER, simde__m512>) {
			if constexpr (Same<T, float>)
				return simde_mm512_set1_ps(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill __m512");
		}
		else if constexpr (Same<REGISTER, simde__m512d>) {
			if constexpr (Same<T, double>)
				return simde_mm512_set1_pd(&value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Fill of __m512d");
		}
		else LANGULUS_ASSERT("Bad REGISTER type for SIMD::Fill");
	}

} // namespace Langulus::TSIMDe

#if LANGULUS_COMPILER(GCC)
	#pragma GCC diagnostic pop
#endif