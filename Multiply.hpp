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
	auto MultiplyInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported{};
	}

	/// Multiply two arrays using SIMD														
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the multiplied elements as a register								
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto MultiplyInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (Integer8<T>) {
				auto loLHS = simde_mm_cvtepi8_epi16(lhs);
				auto loRHS = simde_mm_cvtepi8_epi16(rhs);
				loLHS = simde_mm_mullo_epi16(loLHS, loRHS);

				auto hiLHS = simde_mm_cvtepi8_epi16(_mm_halfflip(lhs));
				auto hiRHS = simde_mm_cvtepi8_epi16(_mm_halfflip(rhs));
				hiLHS = simde_mm_mullo_epi16(hiLHS, hiRHS);

				if constexpr (SignedInteger8<T>)
					return simde_mm_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (Integer16<T>)
				return simde_mm_mullo_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm_mullo_epi32(lhs, rhs);
			else if constexpr (Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_mullo_epi64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm_mul_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (Integer8<T>) {
				auto hiLHS = simde_mm256_unpackhi_epi8(lhs, simde_mm256_setzero_si256());
				auto hiRHS = simde_mm256_unpackhi_epi8(rhs, simde_mm256_setzero_si256());
				hiLHS = simde_mm256_mullo_epi16(hiLHS, hiRHS);

				auto loLHS = simde_mm256_unpacklo_epi8(lhs, simde_mm256_setzero_si256());
				auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm256_setzero_si256());
				loLHS = simde_mm256_mullo_epi16(loLHS, loRHS);

				if constexpr (SignedInteger8<T>)
					return simde_mm256_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm256_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (Integer16<T>)
				return simde_mm256_mullo_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm256_mullo_epi32(lhs, rhs);
			else if constexpr (Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_mullo_epi64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_mul_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (Integer8<T>) {
				auto hiLHS = simde_mm512_unpackhi_epi8(lhs, simde_mm512_setzero_si512());
				auto hiRHS = simde_mm512_unpackhi_epi8(rhs, simde_mm512_setzero_si512());
				hiLHS = simde_mm512_mullo_epi16(hiLHS, hiRHS);

				auto loLHS = simde_mm512_unpacklo_epi8(lhs, simde_mm512_setzero_si512());
				auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm512_setzero_si512());
				loLHS = simde_mm512_mullo_epi16(loLHS, loRHS);

				if constexpr (SignedInteger8<T>)
					return simde_mm512_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm512_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (Integer16<T>)
				return simde_mm512_mullo_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm512_mullo_epi32(lhs, rhs);
			else if constexpr (Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm512_mullo_epi64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_mul_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Multiply(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return MultiplyInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs * rhs;
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Multiply(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Multiply<LHS, RHS>(lhs, rhs);
		if constexpr (TSIMD<decltype(result)>) {
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
	NOD() WRAPPER MultiplyWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Multiply<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD