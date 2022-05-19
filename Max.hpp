#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace PCFW::Math::SIMD
{
		
	template<Number T, pcptr S>
	auto MaxInner(const NotSupported&, const NotSupported&) noexcept {
		return NotSupported{};
	}

	/// Select the bigger values via SIMD													
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - type of register we're operating with					
	///	@param lhs - the left-hand-side array 											
	///	@param rhs - the right-hand-side array 										
	///	@return the maxed values															
	template<Number T, pcptr S, TSIMD REGISTER>
	auto MaxInner(const REGISTER& lhs, const REGISTER& rhs) noexcept {
		if constexpr (SIMD128<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_max_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_max_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm_max_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_max_epu16(lhs, rhs);
			else if constexpr (SignedInteger32<T>)
				return simde_mm_max_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm_max_epu32(lhs, rhs);
			else if constexpr (SignedInteger64<T>) {
				#if LANGULUS_SIMD() >= LANGULUS_SIMD_AVX512()
					return _mm_max_epi64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (UnsignedInteger64<T>) {
				#if LANGULUS_SIMD() >= LANGULUS_SIMD_AVX512()
					return simde_mm_max_epu64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (Same<T, pcr32>)
				return simde_mm_max_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_max_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_max_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_max_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_max_epu16(lhs, rhs);
			else if constexpr (SignedInteger32<T>)
				return simde_mm256_max_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm256_max_epu32(lhs, rhs);
			else if constexpr (SignedInteger64<T>) {
				#if LANGULUS_SIMD() >= LANGULUS_SIMD_AVX512()
					return _mm_max_epi64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (UnsignedInteger64<T>) {
				#if LANGULUS_SIMD() >= LANGULUS_SIMD_AVX512()
					return _mm_max_epu64(lhs, rhs);
				#else
					return SIMD::NotSupported{};
				#endif
			}
			else if constexpr (Same<T, pcr32>)
				return simde_mm256_max_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_max_epi8(lhs, rhs);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_max_epu8(lhs, rhs);
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_max_epi16(lhs, rhs);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_max_epu16(lhs, rhs);
			else if constexpr (SignedInteger32<T>)
				return simde_mm512_max_epi32(lhs, rhs);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_max_epu32(lhs, rhs);
			else if constexpr (SignedInteger64<T>)
				return simde_mm512_max_epi64(lhs, rhs);
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_max_epu64(lhs, rhs);
			else if constexpr (Same<T, pcr32>)
				return simde_mm512_max_ps(lhs, rhs);
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_max_pd(lhs, rhs);
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerMax");
	}

	///																								
	template<Number LHS, Number RHS>
	NOD() auto Max(LHS& lhsOrig, RHS& rhsOrig) noexcept {
		using REGISTER = TRegister<LHS, RHS>;
		using LOSSLESS = TLossless<LHS, RHS>;
		constexpr auto S = ResultSize<LHS, RHS>();
		return AttemptSIMD<0, REGISTER, LOSSLESS>(
			lhsOrig, rhsOrig, 
			[](const REGISTER& lhs, const REGISTER& rhs) noexcept {
				return MaxInner<LOSSLESS, S>(lhs, rhs);
			},
			[](const LOSSLESS& lhs, const LOSSLESS& rhs) noexcept {
				return ::std::max(lhs, rhs);
			}
		);
	}

	///																								
	template<Number LHS, Number RHS, Number OUT>
	void Max(LHS& lhs, RHS& rhs, OUT& output) noexcept {
		const auto result = Max<LHS, RHS>(lhs, rhs);
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
	NOD() WRAPPER MaxWrap(LHS& lhs, RHS& rhs) noexcept {
		WRAPPER result;
		Max<LHS, RHS>(lhs, rhs, result.mArray);
		return result;
	}

} // namespace PCFW::Math::SIMD