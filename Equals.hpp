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

	template<class T, Count S>
	LANGULUS(ALWAYSINLINE) auto EqualsInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}
		
	/// Compare two arrays for equality using SIMD										
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return true if lhs is equal to rhs												
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto EqualsInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
	#if LANGULUS_SIMD(128BIT)
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epi8_mask(lhs, rhs) == 0xFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::UnsignedInteger8<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epu8_mask(lhs, rhs) == 0xFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::SignedInteger16<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epi16_mask(lhs, rhs) == 0xFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::UnsignedInteger16<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epu16_mask(lhs, rhs) == 0xFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::SignedInteger32<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epi32_mask(lhs, rhs) == 0xFF;	// AVX512F + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::UnsignedInteger32<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epu32_mask(lhs, rhs) == 0xFF;	// AVX512F + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(lhs, rhs)) == 0xFFFF; // SSE2
				#endif
			}
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epi64_mask(lhs, rhs) == 0x7;	// AVX512F + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(lhs, rhs)) == 0xFFFF; // SSE4.1
				#endif
			}
			else if constexpr (CT::UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_cmpeq_epu64_mask(lhs, rhs) == 0x7;	// AVX512F + AVX512VL
				#else
					return simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(lhs, rhs)) == 0xFFFF; // SSE4.1
				#endif
			}
			else if constexpr (CT::RealSP<T>)
				return simde_mm_movemask_ps(simde_mm_cmpeq_ps(lhs, rhs)) == 0xF;	// SSE
			else if constexpr (CT::RealDP<T>)
				return simde_mm_movemask_pd(simde_mm_cmpeq_pd(lhs, rhs)) == 0x3;	// SSE2
			else
				LANGULUS_ASSERT("Unsupported type for SIMD::InnerEquals of 16-byte package");
		}
		else
	#endif

	#if LANGULUS_SIMD(256BIT)
		if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epi8_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::UnsignedInteger8<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epu8_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::SignedInteger16<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epi16_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::UnsignedInteger16<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epu16_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512BW + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::SignedInteger32<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epi32_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512F + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::UnsignedInteger32<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epu32_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512F + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epi64_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512F + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_cmpeq_epu64_mask(lhs, rhs) == 0xFFFFFFFF;	// AVX512F + AVX512VL
				#else
					return simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(lhs, rhs)) == 0xFFFFFFFF; // AVX2
				#endif
			}
			else if constexpr (CT::RealSP<T>)
				return simde_mm256_movemask_ps(simde_mm256_cmp_ps(lhs, rhs, _CMP_EQ_OQ)) == 0xFF;	// AVX
			else if constexpr (CT::RealDP<T>)
				return simde_mm256_movemask_pd(simde_mm256_cmp_pd(lhs, rhs, _CMP_EQ_OQ)) == 0xF;	// AVX
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEquals of 32-byte package");
		}
		else
	#endif

	#if LANGULUS_SIMD(512BIT)
		if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm512_cmpeq_epi8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm512_cmpeq_epu8_mask(lhs, rhs) == 0xFFFFFFFFFFFFFFFF;
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm512_cmpeq_epi16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm512_cmpeq_epu16_mask(lhs, rhs) == 0xFFFFFFFF;
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm512_cmpeq_epi32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (CT::UnsignedInteger32<T>)
				return simde_mm512_cmpeq_epu32_mask(lhs, rhs) == 0xFFFF;
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm512_cmpeq_epi64_mask(lhs, rhs) == 0xFF;
			else if constexpr (CT::UnsignedInteger64<T>)
				return simde_mm512_cmpeq_epu64_mask(lhs, rhs) == 0xFF;
			else if constexpr (CT::RealSP<T>)
				return simde_mm512_cmp_ps_mask(lhs, rhs, _CMP_EQ_OQ) == 0xFFFF;
			else if constexpr (CT::RealDP<T>)
				return simde_mm512_cmp_pd_mask(lhs, rhs, _CMP_EQ_OQ) == 0xFF;
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerEquals of 64-byte package");
		}
		else
	#endif

		LANGULUS_ASSERT("Unsupported type for SIMD::InnerEquals");
	}

	/// Compare any lhs and rhs numbers, arrays or not, sparse or dense			
	///	@tparam LHS - left type (deducible)												
	///	@tparam RHS - right type (deducible)											
	///	@param lhsOrig - the left array or number										
	///	@param rhsOrig - the right array or number									
	///	@return true if all elements match												
	template<class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) bool Equals(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();

		if constexpr (S < 2 || CT::NotSupported<REGISTER>) {
			// Call the fallback routine if unsupported or size 1				
			return Fallback<LOSSLESS>(lhsOrig, rhsOrig,
				[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
					return lhs == rhs;
				}
			);
		}
		else {
			const auto result = AttemptSIMD<0, REGISTER, LOSSLESS>(
				lhsOrig, rhsOrig,
				[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
					return EqualsInner<LOSSLESS, S>(lhs, rhs);
				},
				[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
					return lhs == rhs;
				}
			);

			if constexpr (CT::Bool<decltype(result)>) {
				// EqualsInner was called successfully, just return			
				return result;
			}
			else if constexpr (CT::Same<decltype(result), ::std::array<bool, S>>) {
				// Fallback as std::array<bool> - combine							
				for (auto& i : result)
					if (!i) return false;
				return true;
			}
			else LANGULUS_ASSERT("Bad return from AttemptSIMD with EqualsInner");
		}
	}

} // namespace Langulus::SIMD