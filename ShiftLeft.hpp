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
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto ShiftLeftInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm_sllv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm_sllv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m128i");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm256_sllv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm256_sllv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm256_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m256i");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm512_sllv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm512_sllv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm512_sllv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft of __m512i");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftLeft");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto ShiftLeft(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
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
	template<CT::Vector WRAPPER, CT::Number LHS, CT::Number RHS>
	NOD() WRAPPER ShiftLeftWrap(LHS& lhs, RHS& rhs) noexcept {
		const auto result = ShiftLeft<LHS, RHS>(lhs, rhs);
		if constexpr (TSIMD<decltype(result)>) {
			// Extract from register													
			typename WRAPPER::MemberType output[WRAPPER::MemberCount];
			SIMD::Store(result, output);
			return WRAPPER {result};
		}
		else if constexpr (Number<decltype(result)>) {
			// Extract from std::array													
			return WRAPPER {result};
		}
		else {
			// Extract from std::array													
			return WRAPPER {result.data()};
		}
	}

} // namespace Langulus::SIMD