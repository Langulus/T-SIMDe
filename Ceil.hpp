#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace PCFW::Math::SIMD
{

	/// Get ceiling values via SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param value - the array 															
	///	@return the ceiling values															
	template<Number T, pcptr S, TSIMD REGISTER>
	auto InnerCeil(const REGISTER& value) noexcept {
		static_assert(RealNumber<T>,
			"SIMD::InnerFloor is suboptimal and pointless for whole numbers, avoid calling it on such");

		if constexpr (SIMD128<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm_ceil_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm_ceil_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm256_ceil_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_ceil_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (Same<T, pcr32>)
				return simde_mm512_ceil_ps(value);
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_ceil_pd(value);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerCeil");
	}

	template<class T, pcptr S>
	auto Ceil(const T(&value)[S]) noexcept {
		return InnerCeil<T, S>(Load<0>(value));
	}

} // namespace PCFW::Math::SIMD