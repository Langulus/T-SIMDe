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
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

	template<class T, Count S>
	LANGULUS(ALWAYSINLINE) constexpr auto MultiplyInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Multiply two arrays using SIMD														
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the multiplied elements as a register								
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto MultiplyInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::Integer8<T>) {
				auto loLHS = simde_mm_cvtepi8_epi16(lhs);
				auto loRHS = simde_mm_cvtepi8_epi16(rhs);
				loLHS = simde_mm_mullo_epi16(loLHS, loRHS);

				auto hiLHS = simde_mm_cvtepi8_epi16(_mm_halfflip(lhs));
				auto hiRHS = simde_mm_cvtepi8_epi16(_mm_halfflip(rhs));
				hiLHS = simde_mm_mullo_epi16(hiLHS, hiRHS);

				if constexpr (CT::SignedInteger8<T>)
					return simde_mm_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (CT::Integer16<T>)
				return simde_mm_mullo_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm_mullo_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm_mullo_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::RealSP<T>)
				return simde_mm_mul_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::Integer8<T>) {
				auto hiLHS = simde_mm256_unpackhi_epi8(lhs, simde_mm256_setzero_si256());
				auto hiRHS = simde_mm256_unpackhi_epi8(rhs, simde_mm256_setzero_si256());
				hiLHS = simde_mm256_mullo_epi16(hiLHS, hiRHS);

				auto loLHS = simde_mm256_unpacklo_epi8(lhs, simde_mm256_setzero_si256());
				auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm256_setzero_si256());
				loLHS = simde_mm256_mullo_epi16(loLHS, loRHS);

				if constexpr (CT::SignedInteger8<T>)
					return simde_mm256_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm256_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (CT::Integer16<T>)
				return simde_mm256_mullo_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm256_mullo_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm256_mullo_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::RealSP<T>)
				return simde_mm256_mul_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm256_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::Integer8<T>) {
				auto hiLHS = simde_mm512_unpackhi_epi8(lhs, simde_mm512_setzero_si512());
				auto hiRHS = simde_mm512_unpackhi_epi8(rhs, simde_mm512_setzero_si512());
				hiLHS = simde_mm512_mullo_epi16(hiLHS, hiRHS);

				auto loLHS = simde_mm512_unpacklo_epi8(lhs, simde_mm512_setzero_si512());
				auto loRHS = simde_mm256_unpacklo_epi8(rhs, simde_mm512_setzero_si512());
				loLHS = simde_mm512_mullo_epi16(loLHS, loRHS);

				if constexpr (CT::SignedInteger8<T>)
					return simde_mm512_packs_epi16(loLHS, hiLHS);
				else
					return simde_mm512_packus_epi16(loLHS, hiLHS);
			}
			else if constexpr (CT::Integer16<T>)
				return simde_mm512_mullo_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm512_mullo_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>) {
				#if LANGULUS_SIMD(AVX512)
					return _mm512_mullo_epi64(lhs, rhs);
				#else
					return CT::Inner::NotSupported{};
				#endif
			}
			else if constexpr (CT::RealSP<T>)
				return simde_mm512_mul_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm512_mul_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMul");
	}

	///																								
	template<class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) auto Multiply(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept -> REGISTER {
				return MultiplyInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> LOSSLESS {
				return lhs * rhs;
			}
		);
	}

	///																								
	template<class LHS, class RHS, class OUT>
	LANGULUS(ALWAYSINLINE) void Multiply(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Multiply<LHS, RHS>(lhs, rhs);
		if constexpr (CT::TSIMD<decltype(result)>) {
			// Extract from register													
			Store(result, output);
		}
		else if constexpr (!CT::Array<OUT>) {
			// Extract from number														
			output = result;
		}
		else {
			// Extract from std::array													
			std::memcpy(output, result.data(), sizeof(output));
		}
	}

	///																								
	template<CT::Vector WRAPPER, class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) WRAPPER MultiplyWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Multiply<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
