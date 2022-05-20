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
	auto MinInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Select the bigger values via SIMD													
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the maxed values															
	template<CT::Number T, Count S, CT::TSIMD REGISTER>
	auto MinInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm_min_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm_min_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm_min_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm_min_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm_min_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm_min_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_min_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_min_epu64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm_min_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_min_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMin of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm256_min_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm256_min_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm256_min_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm256_min_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm256_min_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm256_min_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_min_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (UnsignedInteger64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_min_epu64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_min_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_min_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMin of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm512_min_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm512_min_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm512_min_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm512_min_epu16(lhs, rhs);
			else if constexpr (CT::SignedInteger32<T>)
				return simde_mm512_min_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_min_epu32(lhs, rhs);
			else if constexpr (CT::SignedInteger64<T>)
				return simde_mm512_min_epi64(lhs, rhs);
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_min_epu64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_min_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_min_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMin of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMin");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Min(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return MinInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return ::std::min(lhs, rhs);
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Min(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Min<LHS, RHS>(lhs, rhs);
		if constexpr (CT::TSIMD<decltype(result)>) {
			// Extract from register													
			SIMD::Store(result, output);
		}
		else if constexpr (CT::Number<decltype(result)>) {
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
	template<CT::Vector WRAPPER, CT::Number LHS, CT::Number RHS>
	NOD() WRAPPER MinWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Min<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD