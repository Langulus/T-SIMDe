///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace Langulus::SIMD
{
		
	/// Get absolute values via SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param v - the array 																
	///	@return the absolute values														
	template<class T, Count S, CT::TSIMD REGISTER>
	auto InnerAbs(const REGISTER& v) noexcept {
		static_assert(Signed<T>, 
			"SIMD::InnerAbs is suboptimal and pointless for unsigned values, avoid calling it on such");

		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm_abs_epi8(v);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm_abs_epi16(v);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm_abs_epi32(v);
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm_abs_epi64(v);
			else if constexpr (Same < T, float> )
				return simde_mm_andnot_ps(simde_mm_set1_ps(-0.0F), v);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_andnot_pd(simde_mm_set1_pd(-0.0F), v);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAbs of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm256_abs_epi8(v);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm256_abs_epi16(v);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm256_abs_epi32(v);
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm256_abs_epi64(v);
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_andnot_ps(simde_mm256_set1_ps(-0.0F), v);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_andnot_pd(simde_mm256_set1_pd(-0.0F), v);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAbs of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm512_abs_epi8(v);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm512_abs_epi16(v);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm512_abs_epi32(v);
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm512_abs_epi64(v);
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_andnot_ps(simde_mm512_set1_ps(-0.0F), v);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_andnot_pd(simde_mm512_set1_pd(-0.0F), v);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAbs of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAbs");
	}

	template<class T, Count S>
	auto Abs(const T(&value)[S]) noexcept {
		return InnerAbs<T, S>(Load<0>(value));
	}

} // namespace Langulus::SIMD