#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace PCFW::Math::SIMD
{

	/// Get floored values via SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param value - the array 															
	///	@return the floored values															
	template<Number T, pcptr S, TSIMD REGISTER>
	auto InnerFloor(const REGISTER& value) noexcept {
		static_assert(RealNumber<T>,
			"SIMD::InnerFloor is suboptimal and pointless for whole numbers, avoid calling it on such");

		if constexpr (SIMD128<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm_floor_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm256_floor_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm512_floor_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_floor_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerFloor");
	}

	template<class T, pcptr S>
	auto Floor(const T(&value)[S]) noexcept {
		return InnerFloor<T, S>(Load<0>(value));
	}

} // namespace PCFW::Math::SIMD