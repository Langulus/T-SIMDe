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
	auto XOrInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// XOr two arrays left using SIMD (shifting in zeroes)							
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the xor'd elements as a register										
	template<CT::Number T, Count S, class REGISTER>
	auto XOrInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::Same<REGISTER,simde__m128i>)
			return simde_mm_xor_si128(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m128>)
			return simde_mm_xor_ps(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m128d>)
			return simde_mm_xor_pd(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m256i>)
			return simde_mm256_xor_si256(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m256>)
			return simde_mm256_xor_ps(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m256d>)
			return simde_mm256_xor_pd(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m512i>)
			return simde_mm512_xor_si512(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m512>)
			return simde_mm512_xor_ps(lhs, rhs);
		else if constexpr (CT::Same<REGISTER,simde__m512d>)
			return simde_mm512_xor_pd(lhs, rhs);
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerXOr");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto XOr(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return XOrInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs ^ rhs;
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void XOr(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = XOr<LHS, RHS>(lhs, rhs);
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
	NOD() WRAPPER XOrWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		XOr<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD