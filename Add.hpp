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
#include "Store.hpp"

namespace Langulus::SIMD
{
		
	template<CT::Number T, Count S>
	auto AddInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Add two arrays using SIMD															
	///	@tparam T - the type of the array element									
	///	@tparam S - the size of the array											
	///	@tparam REGISTER - the register type (deducible)						
	///	@param lhs - the left-hand-side array 										
	///	@param rhs - the right-hand-side array 									
	///	@return the added elements as a register									
	template<CT::Number T, Count S, CT::TSIMD REGISTER>
	auto AddInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (CT::SIMD128<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm_add_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm_adds_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm_add_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm_adds_epu16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm_add_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm_add_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm_add_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 16-byte package");
		}
		else if constexpr (CT::SIMD256<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm256_add_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm256_adds_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm256_add_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm256_adds_epu16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm256_add_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm256_add_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm256_add_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm256_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 32-byte package");
		}
		else if constexpr (CT::SIMD512<REGISTER>) {
			if constexpr (CT::SignedInteger8<T>)
				return simde_mm512_add_epi8(lhs, rhs);
			else if constexpr (CT::UnsignedInteger8<T>)
				return simde_mm512_adds_epu8(lhs, rhs);
			else if constexpr (CT::SignedInteger16<T>)
				return simde_mm512_add_epi16(lhs, rhs);
			else if constexpr (CT::UnsignedInteger16<T>)
				return simde_mm512_adds_epu16(lhs, rhs);
			else if constexpr (CT::Integer32<T>)
				return simde_mm512_add_epi32(lhs, rhs);
			else if constexpr (CT::Integer64<T>)
				return simde_mm512_add_epi64(lhs, rhs);
			else if constexpr (CT::Same<T, float>)
				return simde_mm512_add_ps(lhs, rhs);
			else if constexpr (CT::Same<T, double>)
				return simde_mm512_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd");
	}

	///																								
	template<CT::Number LHS, CT::Number RHS>
	NOD() auto Add(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return AddInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return lhs + rhs;
			}
		);
	}

	///																								
	template<CT::Number LHS, CT::Number RHS, CT::Number OUT>
	void Add(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
		const auto result = Add<LHS, RHS>(lhs, rhs);
		if constexpr (CT::TSIMD<decltype(result)>) {
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
	NOD() WRAPPER AddWrap(const LHS& lhs, const RHS& rhs) noexcept {
		WRAPPER result;
		Add<LHS, RHS>(lhs, rhs, result.mComponents);
		return result;
	}

} // namespace Langulus::SIMD