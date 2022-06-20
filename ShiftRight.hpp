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
	LANGULUS(ALWAYSINLINE) constexpr auto ShiftRightInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Shift two arrays right using SIMD (shifting in zeroes)						
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the shifted elements as a register									
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto ShiftRightInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm_srlv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm_srlv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m128i");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm256_srlv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm256_srlv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm256_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m256i");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::Integer16<T>)
				return simde_mm512_srlv_epi16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm512_srlv_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm512_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m512i");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight");
	}

	///																								
	template<class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) auto ShiftRight(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();

		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return ShiftRightInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept -> LOSSLESS {
				return lhs >> rhs;
			}
		);
	}

	///																								
	template<class LHS, class RHS, class OUT>
	LANGULUS(ALWAYSINLINE) void ShiftRight(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = ShiftRight<LHS, RHS>(lhs, rhs);
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
	NOD() LANGULUS(ALWAYSINLINE) WRAPPER ShiftRightWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		ShiftRight<LHS, RHS>(lhs, rhs, result.mComponents);
		return result;
	}

} // namespace Langulus::SIMD