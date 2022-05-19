#pragma once
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

namespace PCFW::Math::SIMD
{

	class NotSupported {};

	template<class T>
	concept SIMD128 =
		Same<T, simde__m128> || Same<T, simde__m128d> || Same<T, simde__m128i>;

	template<class T>
	concept SIMD256 =
		Same<T, simde__m256> || Same<T, simde__m256d> || Same<T, simde__m256i>;

	template<class T>
	concept SIMD512 =
		Same<T, simde__m512> || Same<T, simde__m512d> || Same<T, simde__m512i>;

	template<class T>
	concept TSIMD = SIMD128<T> || SIMD256<T> || SIMD512<T>;

	template<class T>
	constexpr bool IsNotSupported = Same<T, NotSupported>;

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
	inline pcu8 _mm_hmax_epu8(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu8(vmax, _mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
		vmax = _mm_max_epu8(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epu8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi8(vmax, 0); // SSE4.1
		return reinterpret_cast<const pcu8&>(result); 
	}

	inline pcu16 _mm_hmax_epu16(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu16(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epu16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi16(vmax, 0); // SSE2
		return reinterpret_cast<const pcu16&>(result);
	}

	inline pcu32 _mm_hmax_epu32(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epu32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi32(vmax, 0); // SSE4.1
		return reinterpret_cast<const pcu32&>(result);
	}

	inline pcu64 _mm_hmax_epu64(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epu64(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		#if LANGULUS_ARCHITECTURE_IS(32BIT)
			alignas(16) pcu64 stored[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(stored), v);		// SSE2
			return stored[0];
		#else
			const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
			return reinterpret_cast<const pcu64&>(result);
		#endif
	}

	inline pci8 _mm_hmax_epi8(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 1)); // SSSE3 + SSE2
		vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epi8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi8(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi8(vmax, 0); // SSE4.1
		return reinterpret_cast<const pci8&>(result);
	}

	inline pci16 _mm_hmax_epi16(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 2)); // SSSE3 + SSE2
		vmax = _mm_max_epi16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi16(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		const auto result = _mm_extract_epi16(vmax, 0); // SSE2
		return reinterpret_cast<const pci16&>(result);
	}

	inline pci32 _mm_hmax_epi32(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(1, 2, 3, 0))); // SSE2
		vmax = _mm_max_epi32(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		return _mm_extract_epi32(vmax, 0);	// SSE2
	}

	inline pci64 _mm_hmax_epi64(const __m128i v) noexcept {
		__m128i vmax = v;
		vmax = _mm_max_epi64(vmax, _mm_shuffle_epi32(vmax, _MM_SHUFFLE(2, 3, 0, 1))); // SSE2
		#if LANGULUS_ARCHITECTURE_IS(32BIT)
			alignas(16) pci64 stored[2];
			_mm_store_si128(reinterpret_cast<__m128i*>(stored), v);		// SSE2
			return stored[0];
		#else
			const auto result = _mm_extract_epi64(vmax, 0); // SSE4.1
			return reinterpret_cast<const pcu64&>(result);
		#endif
	}

	/// Fallback OP on a single pair of dense numbers									
	/// It converts LHS and RHS to the most lossless of the two						
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@tparam OP - the operation to invoke (deducible)							
	///	@param lhs - left argument															
	///	@param rhs - right argument														
	///	@param op - the function to invoke												
	///	@return the resulting number or std::array									
	template<class LOSSLESS, Number LHS, Number RHS, class FFALL>
	NOD() auto Fallback(LHS& lhs, RHS& rhs, FFALL&& op)
	requires ::std::invocable<FFALL, LOSSLESS, LOSSLESS> {
		using OUT = ::std::invoke_result_t<FFALL, LOSSLESS, LOSSLESS>;
		if constexpr (pcIsArray<LHS> && pcIsArray<RHS>) {
			// Array OP Array																
			constexpr pcptr S = ((pcExtentOf<LHS>) < (pcExtentOf<RHS>)) ? pcExtentOf<LHS> : pcExtentOf<RHS>;
			::std::array<OUT, S> output;
			for (pcptr i = 0; i < S; ++i)
				output[i] = Fallback<LOSSLESS>(lhs[i], rhs[i], pcForward<decltype(op)>(op));
			return output;
		}
		else if constexpr (pcIsArray<LHS>) {
			// Array OP Scalar															
			constexpr pcptr S = pcExtentOf<LHS>;
			::std::array<OUT, S> output;
			if constexpr (Boolean<OUT>) {
				auto& same_rhs = pcVal(rhs);
				for (pcptr i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(lhs[i], same_rhs, pcForward<decltype(op)>(op));
			}
			else {
				const auto same_rhs = static_cast<LOSSLESS>(pcVal(rhs));
				for (pcptr i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(lhs[i], same_rhs, pcForward<decltype(op)>(op));
			}
			return output;
		}
		else if constexpr (pcIsArray<RHS>) {
			// Scalar OP Array															
			constexpr pcptr S = pcExtentOf<RHS>;
			::std::array<OUT, S> output;
			if constexpr (Boolean<OUT>) {
				auto& same_lhs = pcVal(lhs);
				for (pcptr i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(same_lhs, rhs[i], pcForward<decltype(op)>(op));
			}
			else {
				const auto same_lhs = static_cast<LOSSLESS>(pcVal(lhs));
				for (pcptr i = 0; i < S; ++i)
					output[i] = Fallback<LOSSLESS>(same_lhs, rhs[i], pcForward<decltype(op)>(op));
			}
			return output;
		}
		else {
			// Scalar OP Scalar															
			// Casts should be optimized-out if type is same (I hope)		
			return op(static_cast<LOSSLESS>(pcVal(lhs)), static_cast<LOSSLESS>(pcVal(rhs)));
		}
	}

	/// Constrexpr function to calculate required elements			 				
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@param lhs - left argument															
	///	@param rhs - right argument														
	///	@return the size of the required register										
	template<Number LHS, Number RHS>
	NOD() constexpr pcptr ResultSize() noexcept {
		if constexpr (pcIsArray<LHS> && pcIsArray<RHS>)
			// Array OP Array																
			return pcExtentOf<LHS> < pcExtentOf<RHS> ? pcExtentOf<LHS> : pcExtentOf<RHS>;
		else if constexpr (pcIsArray<LHS>)
			// Array OP Scalar															
			return pcExtentOf<LHS>;
		else if constexpr (pcIsArray<RHS>)
			// Scalar OP Array															
			return pcExtentOf<RHS>;
		else
			// Scalar OP Scalar															
			return 1;
	}

} // namespace PCFW::Math::SIMD