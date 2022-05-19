#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace PCFW::Math::SIMD
{
		
	/// Shift two arrays right using SIMD (shifting in zeroes)						
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the shifted elements as a register									
	template<Number T, pcptr S, TSIMD REGISTER>
	auto ShiftRightInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm_srlv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm_srlv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m128i");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm256_srlv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm256_srlv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm256_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m256i");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (Integer16<T>)
				return simde_mm512_srlv_epi16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm512_srlv_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm512_srlv_epi64(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight of __m512i");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerShiftRight");
	}

	///																								
	template<Number LHS, Number RHS>
	NOD() auto ShiftRight(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return ShiftRightInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs >> rhs;
			}
		);
	}

	///																								
	template<Number LHS, Number RHS, Number OUT>
	void ShiftRight(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = ShiftRight<LHS, RHS>(lhs, rhs);
		if constexpr (TSIMD<decltype(result)>) {
			// Extract from register													
			SIMD::Store(result, output);
		}
		else if constexpr (Number<decltype(result)>) {
			// Extract from number														
			output = result;
		}
		else {
			// Extract from std::array													
			for (pcptr i = 0; i < pcExtentOf<OUT>; ++i)
				output[i] = result[i];
		}
	}

	///																								
	template<ComplexNumber WRAPPER, Number LHS, Number RHS>
	NOD() WRAPPER ShiftRightWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		ShiftRight<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace PCFW::Math::SIMD