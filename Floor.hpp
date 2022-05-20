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
	auto InnerFloor(const REGISTER& value) noexcept {
		static_assert(CT::Real<T>,
			"SIMD::InnerFloor is suboptimal and pointless for whole numbers, avoid calling it on such");

		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm_floor_ps(value);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm256_floor_ps(value);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm512_floor_ps(value);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor");
	}

	template<class T, Count S>
	auto Floor(const T(&value)[S]) noexcept {
		return InnerFloor<T, S>(Load<0>(value));
	}

} // namespace Langulus::SIMD