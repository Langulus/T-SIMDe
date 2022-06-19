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
	LANGULUS(ALWAYSINLINE) auto PowerInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Raise by a power using SIMD															
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the raised values															
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto PowerInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		static_assert(CT::Real<T>, 
			"SIMD::InnerPow doesn't work for whole numbers");

		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm_pow_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm256_pow_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm256_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::RealSP<T>)
				return simde_mm512_pow_ps(lhs, rhs);
			else if constexpr (CT::RealDP<T>)
				return simde_mm512_pow_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerPow");
	}

	///																								
	template<class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) auto Power(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();
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
	template<class LHS, class RHS, class OUT>
	LANGULUS(ALWAYSINLINE) void Power(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Power<LHS, RHS>(lhs, rhs);
		if constexpr (CT::TSIMD<decltype(result)>) {
			// Extract from register													
			Store(result, output);
		}
		else if constexpr (ExtentOf<OUT> == 1) {
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
	template<CT::Vector WRAPPER, class LHS, class RHS>
	NOD() LANGULUS(ALWAYSINLINE) WRAPPER PowerWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Power<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace Langulus::SIMD