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

	/// Get ceiling values via SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param value - the array 															
	///	@return the ceiling values															
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto InnerCeil(const REGISTER& value) noexcept {
		static_assert(CT::Real<T>,
			"SIMD::InnerFloor is suboptimal and pointless for whole numbers, avoid calling it on such");

		#if LANGULUS_SIMD(128BIT)
			if constexpr (CT::SIMD128<REGISTER>) {
				if constexpr (CT::RealSP<T>)
					return simde_mm_ceil_ps(value);
				else if constexpr (CT::RealDP<T>)
					return simde_mm_ceil_pd(value);
				else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 16-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(256BIT)
			if constexpr (CT::SIMD256<REGISTER>) {
				if constexpr (CT::RealSP<T>)
					return simde_mm256_ceil_ps(value);
				else if constexpr (CT::RealDP<T>)
					return simde_mm256_ceil_pd(value);
				else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 32-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(512BIT)
			if constexpr (CT::SIMD512<REGISTER>) {
				if constexpr (CT::RealSP<T>)
					return simde_mm512_ceil_ps(value);
				else if constexpr (CT::RealDP<T>)
					return simde_mm512_ceil_pd(value);
				else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 64-byte package");
			}
			else
		#endif

		LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil");
	}

	template<class T, Count S>
	LANGULUS(ALWAYSINLINE) auto Ceil(const T(&value)[S]) noexcept {
		return InnerCeil<T, S>(Load<0>(value));
	}

} // namespace Langulus::SIMD