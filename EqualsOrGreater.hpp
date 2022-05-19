#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace PCFW::Math::SIMD
{
	
	template<Number T, pcptr S>
	auto EqualsOrGreaterInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported {};
	}

	/// Compare for equal-or-greater two arrays using SIMD							
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return true if lhs is equal-or-greater than rhs							
	template<Number T, pcptr S, TSIMD REGISTER>
	auto EqualsOrGreaterInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_cmpge_epi8_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_cmpge_epu8_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm_cmpge_epi16_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_cmpge_epu16_mask(lhs, rhs) == 0xFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm_cmpge_epi32_mask(lhs, rhs) == 0xF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm_cmpge_epu32_mask(lhs, rhs) == 0xF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm_cmpge_epi64_mask(lhs, rhs) == 0x7;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm_cmpge_epu64_mask(lhs, rhs) == 0x7;
			else if constexpr (Same<T, pcr32>)
				return simde_mm_movemask_ps(_mm_cmpge_ps(lhs, rhs)) == 0xF;
			else if constexpr (Same<T, pcr64>)
				return simde_mm_movemask_pd(_mm_cmpge_pd(lhs, rhs)) == 0x7;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrGreater of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_cmpge_epi8_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_cmpge_epu8_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_cmpge_epi16_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_cmpge_epu16_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm256_cmpge_epi32_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm256_cmpge_epu32_mask(lhs, rhs) == 0xFF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm256_cmpge_epi64_mask(lhs, rhs) == 0xF;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm256_cmpge_epu64_mask(lhs, rhs) == 0xF;
			else if constexpr (Same<T, pcr32>)
				return simde_mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_GE_OQ)) == 0xFF;
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_GE_OQ)) == 0xF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrGreater of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_cmpge_epi8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_cmpge_epu8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_cmpge_epi16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_cmpge_epu16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm512_cmpge_epi32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_cmpge_epu32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm512_cmpge_epi64_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_cmpge_epu64_mask(lhs, rhs) == 0xFF;
			else if constexpr (Same<T, pcr32>)
				return simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_GE_OQ) == 0xFFFF;
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_GE_OQ) == 0xFF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrGreater of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrGreater");
	}

	/// Compare any lhs and rhs numbers, arrays or not, sparse or dense			
	///	@tparam LHS - left type (deducible)												
	///	@tparam RHS - right type (deducible)											
	///	@param lhsOrig - the left array or number										
	///	@param rhsOrig - the right array or number									
	///	@return true if all elements match												
	template<Number LHS, Number RHS>
	NOD() bool EqualsOrGreater(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		const auto result = AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return EqualsOrGreaterInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs >= rhs;
			}
		);

		if constexpr (Boolean<decltype(result)>)
			// EqualsInner was called successfully, just return				
			return result;
		else {
			// Fallback as std::array<bool> - combine								
			for (auto& i : result)
				if (!i) return false;
			return true;
		}
	}

} // namespace PCFW::Math::SIMD