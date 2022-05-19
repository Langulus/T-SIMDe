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
	auto LesserInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported {};
	}

	/// Compare two arrays for lesser using SIMD											
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return true if lhs is lower than rhs											
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto LesserInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			#if LANGULUS_SIMD(AVX512)
				if constexpr (SignedInteger8<T>)
					return _mm_cmplt_epi8_mask(lhs, rhs) == 0xFFFF;
				else if constexpr (UnsignedInteger8<T>)
					return _mm_cmplt_epu8_mask(lhs, rhs) == 0xFFFF;
				else if constexpr (SignedInteger16<T>)
					return _mm_cmplt_epi16_mask(lhs, rhs) == 0xFF;
				else if constexpr (UnsignedInteger16<T>)
					return _mm_cmplt_epu16_mask(lhs, rhs) == 0xFF;
				else if constexpr (SignedInteger32<T>)
					return _mm_cmplt_epi32_mask(lhs, rhs) == 0xF;
				else if constexpr (UnsignedInteger32<T>)
					return _mm_cmplt_epu32_mask(lhs, rhs) == 0xF;
				else if constexpr (SignedInteger64<T>)
					return _mm_cmplt_epi64_mask(lhs, rhs) == 0x7;
				else if constexpr (UnsignedInteger64<T>)
					return _mm_cmplt_epu64_mask(lhs, rhs) == 0x7;
				else if constexpr (CT::Same<T, pcr32>)
					return _mm_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ) == 0xF;
				else if constexpr (CT::Same<T, pcr64>)
					return _mm_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ)) == 0x7;
			#else
				if constexpr (Integer8<T>)
					return simde_mm_movemask_epi8(simde_mm_cmplt_epi8(lhs, rhs)) == 0xFFFF;
				else if constexpr (Integer16<T>)
					return simde_mm_movemask_epi8(simde_mm_cmplt_epi16(lhs, rhs)) == 0xFFFF;
				else if constexpr (Integer32<T>)
					return simde_mm_movemask_epi8(simde_mm_cmplt_epi32(lhs, rhs)) == 0xFFFF;
				else if constexpr (Integer64<T>)
					return NotSupported{};
				else if constexpr (CT::Same<T, float>)
					return simde_mm_movemask_ps(_mm_cmplt_ps(lhs, rhs)) == 0xF;
				else if constexpr (CT::Same<T, double>)
					return simde_mm_movemask_pd(_mm_cmplt_pd(lhs, rhs)) == 0x7;
			#endif
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLower of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			#if LANGULUS_SIMD(AVX512)
				if constexpr (SignedInteger8<T>)
					return _mm256_cmplt_epi8_mask(lhs, rhs) == 0xFFFFFFFF;
				else if constexpr (UnsignedInteger8<T>)
					return _mm256_cmplt_epu8_mask(lhs, rhs) == 0xFFFFFFFF;
				else if constexpr (SignedInteger16<T>)
					return _mm256_cmplt_epi16_mask(lhs, rhs) == 0xFFFF;
				else if constexpr (UnsignedInteger16<T>)
					return _mm256_cmplt_epu16_mask(lhs, rhs) == 0xFFFF;
				else if constexpr (SignedInteger32<T>)
					return _mm256_cmplt_epi32_mask(lhs, rhs) == 0xFF;
				else if constexpr (UnsignedInteger32<T>)
					return _mm256_cmplt_epu32_mask(lhs, rhs) == 0xFF;
				else if constexpr (SignedInteger64<T>)
					return _mm256_cmplt_epi64_mask(lhs, rhs) == 0xF;
				else if constexpr (UnsignedInteger64<T>)
					return _mm256_cmplt_epu64_mask(lhs, rhs) == 0xF;
				else if constexpr (CT::Same<T, pcr32>)
					return _mm256_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ) == 0xFF;
				else if constexpr (CT::Same<T, pcr64>)
					return _mm256_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ)) == 0xF;
				else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLower of 32-byte package");
			#else
				if constexpr (Integer8<T>)
					return simde_mm256_movemask_epi8(simde_mm256_cmpgt_epi8(lhs, rhs)) == 0 &&
							 simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(lhs, rhs)) == 0;
				else if constexpr (Integer16<T>)
					return simde_mm256_movemask_epi8(simde_mm256_cmpgt_epi16(lhs, rhs)) == 0 &&
							 simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(lhs, rhs)) == 0;
				else if constexpr (Integer32<T>)
					return simde_mm256_movemask_epi8(simde_mm256_cmpgt_epi32(lhs, rhs)) == 0 &&
							 simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(lhs, rhs)) == 0;
				else if constexpr (Integer64<T>)
					return simde_mm256_movemask_epi8(simde_mm256_cmpgt_epi64(lhs, rhs)) == 0 &&
							 simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(lhs, rhs)) == 0;
				else if constexpr (CT::Same<T, float>)
					return simde_mm256_movemask_ps(simde_mm256_cmp_ps(lhs, rhs, _CMP_LT_OQ)) == 0xFF;
				else if constexpr (CT::Same<T, double>)
					return simde_mm256_movemask_pd(simde_mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ)) == 0xF;
			#endif
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLower of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_cmplt_epi8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_cmplt_epu8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_cmplt_epi16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_cmplt_epu16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (SignedInteger32<T>)
				return simde_mm512_cmplt_epi32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_cmplt_epu32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (SignedInteger64<T>)
				return simde_mm512_cmplt_epi64_mask(lhs, rhs) == 0xFF;
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_cmplt_epu64_mask(lhs, rhs) == 0xFF;
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_LT_OQ) == 0xFFFF;
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_LT_OQ) == 0xFF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLower of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLower");
	}

	/// Compare any lhs and rhs numbers, arrays or not, sparse or dense			
	///	@tparam LHS - left type (deducible)												
	///	@tparam RHS - right type (deducible)											
	///	@param lhsOrig - the left array or number										
	///	@param rhsOrig - the right array or number									
	///	@return true if all elements match												
	template<CT::Number LHS, CT::Number RHS>
	NOD() bool Lesser(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		const auto result = AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return LesserInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs < rhs;
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

} // namespace Langulus::SIMD