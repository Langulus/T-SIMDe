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
	auto SubtractInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported{};
	}

	/// Subtract two arrays using SIMD														
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the subtracted elements as a register								
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto SubtractInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_sub_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_subs_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm_sub_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_subs_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm_sub_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm_sub_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm_sub_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_sub_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Sub of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_sub_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_subs_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_sub_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_subs_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm256_sub_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm256_sub_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_sub_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_sub_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Sub of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_sub_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_subs_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_sub_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_subs_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm512_sub_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm512_sub_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_sub_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_sub_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::Sub of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::Sub");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Subtract(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return SubtractInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs - rhs;
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Subtract(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Subtract<LHS, RHS>(lhs, rhs);
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
	NOD() WRAPPER SubtractWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Subtract<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD