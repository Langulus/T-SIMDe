///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include <Langulus.Core.hpp>

/// Make sure everything SIMDe includes is included before SIMDe itself,		
/// so that we	can capsulate it in our namespace later, without encapsulating	
/// std stuff																						
#include <simde/simde-common.h>
#include <cstdint>
#include <type_traits>
#include <bit>
#include <utility>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <complex>

#if defined(SIMDE_X86_MMX_NATIVE)
	#define SIMDE_X86_MMX_USE_NATIVE_TYPE
#elif defined(SIMDE_X86_SSE_NATIVE)
	#define SIMDE_X86_MMX_USE_NATIVE_TYPE
#endif

#if defined(SIMDE_X86_MMX_USE_NATIVE_TYPE)
	#include <mmintrin.h>
#elif defined(SIMDE_ARM_NEON_A32V7_NATIVE)
	#include <arm_neon.h>
#elif defined(SIMDE_MIPS_LOONGSON_MMI_NATIVE)
	#include <loongson-mmiintrin.h>
#endif

#include <stdint.h>
#include <limits.h>

#if defined(_WIN32) && !defined(SIMDE_X86_SSE_NATIVE) && defined(_MSC_VER)
	#include <windows.h>
#endif

#if defined(__ARM_ACLE)
	#include <arm_acle.h>
#endif

namespace Langulus::SIMD
{

	template<class T1, class T2>
	concept Same = CT::Same<T1, T2>;

	#include <simde/x86/avx512.h>
	#include <simde/x86/avx2.h>
	#include <simde/x86/avx.h>
	#include <simde/x86/sse4.2.h>
	#include <simde/x86/sse4.1.h>
	#include <simde/x86/ssse3.h>
	#include <simde/x86/sse3.h>
	#include <simde/x86/sse2.h>
	#include <simde/x86/sse.h>
	#include <simde/x86/svml.h>

	class NotSupported {};

	template<class T>
	concept SIMD128 =
		CT::Same<T, simde__m128> || CT::Same<T, simde__m128d> || CT::Same<T, simde__m128i>;

	template<class T>
	concept SIMD256 =
		CT::Same<T, simde__m256> || CT::Same<T, simde__m256d> || CT::Same<T, simde__m256i>;

	template<class T>
	concept SIMD512 =
		CT::Same<T, simde__m512> || CT::Same<T, simde__m512d> || CT::Same<T, simde__m512i>;

	template<class T>
	concept TSIMD = SIMD128<T> || SIMD256<T> || SIMD512<T>;

	template<class T>
	constexpr bool IsNotSupported = CT::Same<T, NotSupported>;

	///																								
	inline simde__m128 _mm_halfflip(const simde__m128& what) noexcept {
		return simde_mm_permute_ps(what, _MM_SHUFFLE(2, 3, 0, 1));	// AVX
	}

	inline simde__m128d _mm_halfflip(const simde__m128d& what) noexcept {
		return simde_mm_permute_pd(what, _MM_SHUFFLE(1, 0, 0, 0));	// AVX
	}

	inline simde__m128i _mm_halfflip(const simde__m128i& what) noexcept {
		return simde_mm_shuffle_epi32(what, _MM_SHUFFLE(0, 1, 2, 3));	// SSE2
	}

	inline simde__m256 _mm_halfflip(const simde__m256& what) noexcept {
		return simde_mm256_permute2f128_ps(what, what, 0x20);	// AVX
	}

	inline simde__m256d _mm_halfflip(const simde__m256d& what) noexcept {
		return simde_mm256_permute2f128_pd(what, what, 0x20);	// AVX
	}

	inline simde__m256i _mm_halfflip(const simde__m256i& what) noexcept {
		return simde_mm256_permute2x128_si256(what, what, 0x20);	// AVX2
	}

	inline simde__m512 _mm_halfflip(const simde__m512& what) noexcept {
		return simde_mm512_shuffle_f32x4(what, what, _MM_SHUFFLE(2, 3, 0, 1));	// AVX512F
	}

	inline simde__m512d _mm_halfflip(const simde__m512d& what) noexcept {
		return simde_mm512_shuffle_f64x2(what, what, _MM_SHUFFLE(2, 3, 0, 1));	// AVX512F
	}

	inline simde__m512i _mm_halfflip(const simde__m512i& what) noexcept {
		return simde_mm512_shuffle_i64x2(what, what, _MM_SHUFFLE(2, 3, 0, 1));	// AVX512F
	}

	///																								
	inline uint8_t _mm_hmax_epu8(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu8(vmax, _mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
		vmax = _mm_max_epu8(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epu8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi8(vmax, 0); // SSE4.1
		return reinterpret_cast<const uint8_t&>(result);
	}

	inline uint16_t _mm_hmax_epu16(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu16(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epu16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi16(vmax, 0); // SSE2
		return reinterpret_cast<const uint16_t&>(result);
	}

	inline uint32_t _mm_hmax_epu32(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi32(vmax, 0); // SSE4.1
		return reinterpret_cast<const uint32_t&>(result);
	}

	inline uint64_t _mm_hmax_epu64(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu64(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		#if LANGULUS_BITNESS() == 32
			alignas(16) pcu64 stored[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(stored), v);		// SSE2
			return stored[0];
		#else
			const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
			return reinterpret_cast<const uint64_t&>(result);
		#endif
	}

	inline int8_t _mm_hmax_epi8(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
		vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epi8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi8(vmax, 0); // SSE4.1
		return reinterpret_cast<const int8_t&>(result);
	}

	inline int16_t _mm_hmax_epi16(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epi16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi16(vmax, 0); // SSE2
		return reinterpret_cast<const int16_t&>(result);
	}

	inline int32_t _mm_hmax_epi32(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		return _mm_extract_epi32(vmax, 0);	// SSE2
	}

	inline int64_t _mm_hmax_epi64(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi64(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		#if LANGULUS_BITNESS() == 32
			alignas(16) pci64 stored[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(stored), v);		// SSE2
			return stored[0];
		#else
			const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
			return reinterpret_cast<const int64_t&>(result);
		#endif
	}

	template<class LOSSLESS, class FFALL>
	concept Invocable = ::std::invocable<FFALL, LOSSLESS, LOSSLESS>;

	/// Fallback OP on a single pair of dense numbers									
	/// It converts LHS and RHS to the most lossless of the two						
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@tparam FFALL - the operation to invoke on fallback (deducible)		
	///	@param lhs - left argument															
	///	@param rhs - right argument														
	///	@param op - the fallback function to invoke									
	///	@return the resulting number or std::array									
	template<class LOSSLESS, CT::Number LHS, CT::Number RHS, class FFALL>
	NOD() auto Fallback(LHS& lhs, RHS& rhs, FFALL&& op) requires Invocable<FFALL, LOSSLESS> {
		using OUT = ::std::invoke_result_t<FFALL, LOSSLESS, LOSSLESS>;
		if constexpr (CT::Array<LHS> && CT::Array<RHS>) {
			// Array OP Array																
			constexpr auto S = ((ExtentOf<LHS>) < (ExtentOf<RHS>)) ? ExtentOf<LHS> : ExtentOf<RHS>;
			::std::array<OUT, S> output;
			for (Count i = 0; i < S; ++i)
				output[i] = Fallback<LOSSLESS>(lhs[i], rhs[i], Forward<decltype(op)>(op));
			return output;
		}
		else if constexpr (CT::Array<LHS>) {
			// Array OP Scalar															
			constexpr auto S = ExtentOf<LHS>;
			::std::array<OUT, S> output;
			if constexpr (CT::Bool<OUT>) {
				auto& same_rhs = MakeDense(rhs);
				for (Count i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(lhs[i], same_rhs, Forward<decltype(op)>(op));
			}
			else {
				const auto same_rhs = static_cast<LOSSLESS>(MakeDense(rhs));
				for (Count i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(lhs[i], same_rhs, Forward<decltype(op)>(op));
			}
			return output;
		}
		else if constexpr (CT::Array<RHS>) {
			// Scalar OP Array															
			constexpr auto S = ExtentOf<RHS>;
			::std::array<OUT, S> output;
			if constexpr (CT::Bool<OUT>) {
				auto& same_lhs = MakeDense(lhs);
				for (Count i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(same_lhs, rhs[i], Forward<decltype(op)>(op));
			}
			else {
				const auto same_lhs = static_cast<LOSSLESS>(pcVal(lhs));
				for (Count i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(same_lhs, rhs[i], Forward<decltype(op)>(op));
			}
			return output;
		}
		else {
			// Scalar OP Scalar															
			// Casts should be optimized-out if type is same (I hope)		
			return op(
				static_cast<LOSSLESS>(MakeDense(lhs)), 
				static_cast<LOSSLESS>(MakeDense(rhs))
			);
		}
	}

	/// Constrexpr function to calculate required elements			 				
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@return the overlapping count of LHS and RHS									
	template<CT::Number LHS, CT::Number RHS>
	NOD() constexpr Count ResultCount() noexcept {
		if constexpr (CT::Array<LHS> && CT::Array<RHS>)
			// Array OP Array																
			return ExtentOf<LHS> < ExtentOf<RHS> ? ExtentOf<LHS> : ExtentOf<RHS>;
		else if constexpr (CT::Array<LHS>)
			// Array OP Scalar															
			return ExtentOf<LHS>;
		else if constexpr (CT::Array<RHS>)
			// Scalar OP Array															
			return ExtentOf<RHS>;
		else
			// Scalar OP Scalar															
			return 1;
	}

} // namespace Langulus::TSIMDe