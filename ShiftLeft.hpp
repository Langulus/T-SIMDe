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
		
	/// Shift two arrays left using SIMD (shifting in zeroes)						
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the shifted elements as a register									
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto ShiftLeftInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm_sllv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm_sllv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m128i");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm256_sllv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm256_sllv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm256_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m256i");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm512_sllv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm512_sllv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm512_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m512i");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft");
	}

	///																								
	template<class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) auto ShiftLeft(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return ShiftLeftInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs << rhs;
			}
		);
	}

	///																								
	template<CT::Vector WRAPPER, class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) WRAPPER ShiftLeftWrap(LHS& lhs, RHS& rhs) noexcept {
		const auto result = ShiftLeft<LHS, RHS>(lhs, rhs);

		if constexpr (CT::TSIMD<decltype(result)>) {
			// Extract from register													
			typename WRAPPER::MemberType output[WRAPPER::MemberCount];
			Store(result, output);
			return WRAPPER {result};
		}
		else if constexpr (CT::Number<decltype(result)>) {
			// Extract from std::array													
			return WRAPPER {result};
		}
		else {
			// Extract from std::array													
			return WRAPPER {result.data()};
		}
	}

} // namespace Langulus::SIMD