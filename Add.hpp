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
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{
		
	template<class T, Count S>
	LANGULUS(ALWAYSINLINE) auto AddInner(const CT::Inner::NotSupported&, const CT::Inner::NotSupported&) noexcept {
		return CT::Inner::NotSupported{};
	}

	/// Add two arrays using SIMD																
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the added elements as a register										
	template<class T, Count S, CT::TSIMD REGISTER>
	LANGULUS(ALWAYSINLINE) auto AddInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		#if LANGULUS_SIMD(128BIT)
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
				else if constexpr (CT::RealSP<T>)
					return simde_mm_add_ps(lhs, rhs);
				else if constexpr (CT::RealDP<T>)
					return simde_mm_add_pd(lhs, rhs);
				else
					LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 16-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(256BIT)
			if constexpr (CT::SIMD256<REGISTER>) {
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
				else if constexpr (CT::RealSP<T>)
					return simde_mm256_add_ps(lhs, rhs);
				else if constexpr (CT::RealDP<T>)
					return simde_mm256_add_pd(lhs, rhs);
				else
					LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 32-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(512BIT)
			if constexpr (CT::SIMD512<REGISTER>) {
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
				else if constexpr (CT::RealSP<T>)
					return simde_mm512_add_ps(lhs, rhs);
				else if constexpr (CT::RealDP<T>)
					return simde_mm512_add_pd(lhs, rhs);
				else
					LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 64-byte package");
			}
			else
		#endif

		LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd");
	}

	///																								
	template<class LHS, class RHS>
	LANGULUS(ALWAYSINLINE) NOD() auto Add(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
		using REGISTER = CT::Register<LHS, RHS>;
		using LOSSLESS = CT::Lossless<LHS, RHS>;
		constexpr auto S = OverlapCount<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return AddInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				if constexpr (CT::Same<LOSSLESS, ::std::byte>) {
					// ::std::byte doesn't have + operator							
					return static_cast<LOSSLESS>(
						reinterpret_cast<const unsigned char&>(lhs) +
						reinterpret_cast<const unsigned char&>(rhs)
					);
				}
				else return lhs + rhs;
			}
		);
	}

	///																								
	template<class LHS, class RHS, class OUT>
	LANGULUS(ALWAYSINLINE) void Add(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
		const auto result = Add<LHS, RHS>(lhs, rhs);
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
	LANGULUS(ALWAYSINLINE) NOD() WRAPPER AddWrap(const LHS& lhs, const RHS& rhs) noexcept {
		WRAPPER result;
		Add<LHS, RHS>(lhs, rhs, result.mComponents);
		return result;
	}

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
