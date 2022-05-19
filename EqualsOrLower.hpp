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
	
	template<CT::Number T, Count S>
	auto EqualsOrLesserInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported {};
	}

	/// Compare for equal-or-greater two arrays using SIMD							
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return true if lhs is equal-or-greater than rhs							
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto EqualsOrLesserInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_cmple_epi8_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_cmple_epu8_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm_cmple_epi16_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_cmple_epu16_mask(lhs, rhs) == 0xFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm_cmple_epi32_mask(lhs, rhs) == 0xF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm_cmple_epu32_mask(lhs, rhs) == 0xF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm_cmple_epi64_mask(lhs, rhs) == 0x7;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm_cmple_epu64_mask(lhs, rhs) == 0x7;
			else if constexpr (Same<T, float>)
				return simde_mm_movemask_ps(_mm_cmple_ps(lhs, rhs)) == 0xF;
			else if constexpr (Same<T, double>)
				return simde_mm_movemask_pd(_mm_cmple_pd(lhs, rhs)) == 0x7;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrLower of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_cmple_epi8_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_cmple_epu8_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_cmple_epi16_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_cmple_epu16_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm256_cmple_epi32_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm256_cmple_epu32_mask(lhs, rhs) == 0xFF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm256_cmple_epi64_mask(lhs, rhs) == 0xF;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm256_cmple_epu64_mask(lhs, rhs) == 0xF;
			else if constexpr (Same<T, float>)
				return simde_mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_LE_OQ)) == 0xFF;
			else if constexpr (Same<T, double>)
				return simde_mm256_movemask_pd(_mm256_cmp_pd(lhs, rhs, _CMP_LE_OQ)) == 0xF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrLower of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_cmple_epi8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_cmple_epu8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_cmple_epi16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_cmple_epu16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm512_cmple_epi32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_cmple_epu32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm512_cmple_epi64_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_cmple_epu64_mask(lhs, rhs) == 0xFF;
			else if constexpr (Same<T, float>)
				return simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_LE_OQ) == 0xFFFF;
			else if constexpr (Same<T, double>)
				return simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_LE_OQ) == 0xFF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrLower of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEqualsOrLower");
	}

	/// Compare any lhs and rhs numbers, arrays or not, sparse or dense			
	///	@tparam LHS - left type (deducible)												
	///	@tparam RHS - right type (deducible)											
	///	@param lhsOrig - the left array or number										
	///	@param rhsOrig - the right array or number									
	///	@return true if all elements match												
	template<CT::Number LHS, CT::Number RHS>
	NOD() bool EqualsOrLesser(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		const auto result = AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return EqualsOrLesserInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs <= rhs;
			}
		);

		if constexpr (CT::Bool<decltype(result)>)
			// EqualsInner was called successfully, just return				
			return result;
		else {
			// Fallback as std::array<bool> - combine								
			for (auto& i : result)
				if (!i) return false;
			return true;
		}
	}

} // namespace Langulus::TSIMDe