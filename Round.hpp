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

	/// Get floored values via SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param value - the array 															
	///	@return the floored values															
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto InnerRound(const REGISTER& value) noexcept {
		static_assert(CT::Real<T>,
			"SIMD::InnerFloor is suboptimal for unreal numbers, avoid calling it on such");

		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else if constexpr (CT::RealDP<T>)
				return simde_mm_round_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerRound of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm256_round_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else if constexpr (CT::RealDP<T>)
				return simde_mm256_round_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerRound of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm512_roundscale_ps(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else if constexpr (CT::RealDP<T>)
				return simde_mm512_roundscale_pd(value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerRound of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerRound");
	}

	template<class T, Count S>
	LANGULUS(ALWAYSINLINE) auto Round(const T(&value)[S]) noexcept {
		return InnerRound<T, S>(Load<0>(value));
	}

} // namespace Langulus::SIMD