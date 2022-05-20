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
	auto PowerInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Raise by a power using SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the raised values															
	template<CT::Number T, Count S, CT::TSIMD REGISTER>
	auto PowerInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		static_assert(CT::Real<T>, 
			"SIMD::InnerPow doesn't work for whole numbers");

		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm_pow_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm256_pow_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::Same<T, float>)
				return simde_mm512_pow_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Power(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return PowerInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return ::std::pow(lhs, rhs);
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Power(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Power<LHS, RHS>(lhs, rhs);
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
	NOD() WRAPPER PowerWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Power<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD