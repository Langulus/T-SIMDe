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
	auto MaxInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Select the bigger values via SIMD													
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the maxed values															
	template<class T, Count S, CT::TSIMD REGISTER>
	auto MaxInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm_max_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm_max_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm_max_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm_max_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm_max_epi32(lhs, rhs);
			else if constexpr (CT::UnsignedInteger32<T>)
				return simde_mm_max_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_max_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return simde_mm_max_epu64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm_max_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm256_max_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm256_max_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm256_max_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm256_max_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm256_max_epi32(lhs, rhs);
			else if constexpr (CT::UnsignedInteger32<T>)
				return simde_mm256_max_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_max_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_max_epu64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_max_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm512_max_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm512_max_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm512_max_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm512_max_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm512_max_epi32(lhs, rhs);
			else if constexpr (CT::UnsignedInteger32<T>)
				return simde_mm512_max_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm512_max_epi64(lhs, rhs);
			else if constexpr (CT::UnsignedInteger64<T>)
				return simde_mm512_max_epu64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_max_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax");
	}

	///																								
	template<class LHS, class RHS>
	NOD() auto Max(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return MaxInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return ::std::max(lhs, rhs);
			}
		);
	}

	///																								
	template<class LHS, class RHS, class OUT>
	void Max(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Max<LHS, RHS>(lhs, rhs);
		if constexpr (CT::TSIMD<decltype(result)>) {
			// Extract from register													
			Store(result, output);
		}
		else if constexpr (ExtentOf<OUT> == 1) {
			// Extract from number														
			output = result;
		}
		else {
			// Extract from std::array													
			for (Offset i = 0; i < ExtentOf<OUT>; ++i)
				output[i] = result[i];
		}
	}

	///																								
	template<CT::Vector WRAPPER, class LHS, class RHS>
	NOD() WRAPPER MaxWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Max<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD