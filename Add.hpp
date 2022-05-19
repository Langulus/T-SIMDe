#pragma once
#include "Fill.hpp"
#include "Convert.hpp"
#include "Store.hpp"

namespace PCFW::Math::SIMD
{
		
	template<Number T, pcptr S>
	auto AddInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported{};
	}

	/// Add two arrays using SIMD															
	///	@tparam T - the type of the array element									
	///	@tparam S - the size of the array											
	///	@tparam REGISTER - the register type (deducible)						
	///	@param lhs - the left-hand-side array 										
	///	@param rhs - the right-hand-side array 									
	///	@return the added elements as a register									
	template<Number T, pcptr S, TSIMD REGISTER>
	auto AddInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_add_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_adds_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm_add_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_adds_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm_add_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm_add_epi64(lhs, rhs);
			else if constexpr (Same<T, pcr32>)
				return simde_mm_add_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_add_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_adds_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_add_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_adds_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm256_add_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm256_add_epi64(lhs, rhs);
			else if constexpr (Same<T, pcr32>)
				return simde_mm256_add_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_add_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_adds_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_add_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_adds_epu16(lhs, rhs);
			else if constexpr (Integer32<T>)
				return simde_mm512_add_epi32(lhs, rhs);
			else if constexpr (Integer64<T>)
				return simde_mm512_add_epi64(lhs, rhs);
			else if constexpr (Same<T, pcr32>)
				return simde_mm512_add_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_add_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerAdd");
	}

	///																								
	template<Number LHS, Number RHS>
	NOD() auto Add(const LHS& lhsOrig, const RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
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
	template<Number LHS, Number RHS, Number OUT>
	void Add(const LHS& lhs, const RHS& rhs, OUT& output) noexcept {
		const auto result = Add<LHS, RHS>(lhs, rhs);
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
	NOD() WRAPPER AddWrap(const LHS& lhs, const RHS& rhs) noexcept {
		WRAPPER result;
		Add<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace PCFW::Math::SIMD