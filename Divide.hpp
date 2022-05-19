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
	auto DivideInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported{};
	}

	/// Divide two arrays using SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the divided elements as a register									
	template<CT::Number T, Count S, TSIMD REGISTER>
	auto DivideInner(const REGISTER& lhs, const REGISTER& rhs) {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (UnsignedInteger8<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epu8(lhs, rhs);
			}
			else if constexpr (SignedInteger8<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epi8(lhs, rhs);
			}
			else if constexpr (UnsignedInteger16<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epu16(lhs, rhs);
			}
			else if constexpr (SignedInteger16<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi16(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epi16(lhs, rhs);
			}
			else if constexpr (UnsignedInteger32<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epu32(lhs, rhs);
			}
			else if constexpr (SignedInteger32<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi32(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epi32(lhs, rhs);
			}
			else if constexpr (UnsignedInteger64<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epu64(lhs, rhs);
			}
			else if constexpr (SignedInteger64<T>) {
				if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi64(rhs, simde_mm_setzero_si128())))
					throw Except::DivisionByZero();
				return simde_mm_div_epi64(lhs, rhs);
			}
			else if constexpr (Same<T, float>) {
				if (simde_mm_movemask_ps(simde_mm_cmpeq_ps(rhs, simde_mm_setzero_ps())))
					throw Except::DivisionByZero();
				return simde_mm_div_ps(lhs, rhs);
			}
			else if constexpr (Same<T, double>) {
				if (simde_mm_movemask_pd(simde_mm_cmpeq_pd(rhs, simde_mm_setzero_pd())))
					throw Except::DivisionByZero();
				return simde_mm_div_pd(lhs, rhs);
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerDiv of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (UnsignedInteger8<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epu8(lhs, rhs);
			}
			else if constexpr (SignedInteger8<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epi8(lhs, rhs);
			}
			else if constexpr (UnsignedInteger16<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epu16(lhs, rhs);
			}
			else if constexpr (SignedInteger16<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi16(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epi16(lhs, rhs);
			}
			else if constexpr (UnsignedInteger32<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epu32(lhs, rhs);
			}
			else if constexpr (SignedInteger32<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi32(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epi32(lhs, rhs);
			}
			else if constexpr (UnsignedInteger64<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epu64(lhs, rhs);
			}
			else if constexpr (SignedInteger64<T>) {
				if (simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi64(rhs, simde_mm256_setzero_si256())))
					throw Except::DivisionByZero();
				return simde_mm256_div_epi64(lhs, rhs);
			}
			else if constexpr (Same<T, float>) {
				if (simde_mm256_movemask_ps(simde_mm256_cmp_ps(rhs, simde_mm256_setzero_ps(), _CMP_EQ_OQ)))
					throw Except::DivisionByZero();
				return simde_mm256_div_ps(lhs, rhs);
			}
			else if constexpr (Same<T, double>) {
				if (simde_mm256_movemask_pd(simde_mm256_cmp_pd(rhs, simde_mm256_setzero_pd(), _CMP_EQ_OQ)))
					throw Except::DivisionByZero();
				return simde_mm256_div_pd(lhs, rhs);
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerDiv of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (UnsignedInteger8<T>) {
				if (simde_mm512_cmpeq_epi8(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epu8(lhs, rhs);
			}
			else if constexpr (SignedInteger8<T>) {
				if (simde_mm512_cmpeq_epi8(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epi8(lhs, rhs);
			}
			else if constexpr (UnsignedInteger16<T>) {
				if (simde_mm512_cmpeq_epi16(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epu16(lhs, rhs);
			}
			else if constexpr (SignedInteger16<T>) {
				if (simde_mm512_cmpeq_epi16(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epi16(lhs, rhs);
			}
			else if constexpr (UnsignedInteger32<T>) {
				if (simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epu32(lhs, rhs);
			}
			else if constexpr (SignedInteger32<T>) {
				if (simde_mm512_cmpeq_epi32(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epi32(lhs, rhs);
			}
			else if constexpr (UnsignedInteger64<T>) {
				if (simde_mm512_cmpeq_epi64(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epu64(lhs, rhs);
			}
			else if constexpr (SignedInteger64<T>) {
				if (simde_mm512_cmpeq_epi64(rhs, simde_mm512_setzero_si512()))
					throw Except::DivisionByZero();
				return simde_mm512_div_epi64(lhs, rhs);
			}
			else if constexpr (Same<T, float>) {
				if (simde_mm512_cmp_ps(rhs, simde_mm512_setzero_ps(), _CMP_EQ_OQ))
					throw Except::DivisionByZero();
				return simde_mm512_div_ps(lhs, rhs);
			}
			else if constexpr (Same<T, double>) {
				if (simde_mm512_cmp_pd(rhs, simde_mm512_setzero_pd(), _CMP_EQ_OQ))
					throw Except::DivisionByZero();
				return simde_mm512_div_pd(lhs, rhs);
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerDiv of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerDiv");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Divide(LHS& lhsOrig, RHS& rhsOrig) {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<1, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) {
				return DivideInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) {
				if (rhs == 0)
					throw Except::DivisionByZero();
				return lhs / rhs;
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Divide(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Divide<LHS, RHS>(lhs, rhs);
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
	template<ComplexNumber WRAPPER, CT::Number LHS, CT::Number RHS>
	NOD() WRAPPER DivideWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Divide<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::TSIMDe