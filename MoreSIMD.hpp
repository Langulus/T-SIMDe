///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Intrinsics.hpp"

namespace Langulus::SIMD
{
	/// The credit for these goes to this wonderful guy:                          
	/// https://github.com/aklomp/sse-intrinsics-tests                            

	/// Calculate absolute difference: abs(x - y)											
	inline simde__m128i _mm_absdiff_epu16(simde__m128i x, simde__m128i y) {
		return simde_mm_or_si128(simde_mm_subs_epu16(x, y), simde_mm_subs_epu16(y, x));
	}

	inline simde__m128i _mm_absdiff_epu8(simde__m128i x, simde__m128i y) {
		return simde_mm_or_si128(simde_mm_subs_epu8(x, y), simde_mm_subs_epu8(y, x));
	}

	/// Replace bit in x with bit in y when matching bit in mask is set				
	inline simde__m128i _mm_blendv_si128(simde__m128i x, simde__m128i y, simde__m128i mask) {
		return simde_mm_or_si128(simde_mm_andnot_si128(mask, x), simde_mm_and_si128(mask, y));
	}

	/// Swap upper and higher byte in each 16-bit word										
	inline simde__m128i _mm_bswap_epi16(simde__m128i x) {
		return simde_mm_or_si128(simde_mm_slli_epi16(x, 8), simde_mm_srli_epi16(x, 8));
	}

	/// Reverse order of bytes in each 32-bit word											
	inline simde__m128i _mm_bswap_epi32(simde__m128i x) {
#ifdef __SSSE3__
		return simde_mm_shuffle_epi8(x, simde_mm_set_epi8(
			12, 13, 14, 15,
			8, 9, 10, 11,
			4, 5, 6, 7,
			0, 1, 2, 3
		));
#else
		// First swap bytes in each 16-bit word									
		simde__m128i a = simde_mm_or_si128(simde_mm_slli_epi16(x, 8), simde_mm_srli_epi16(x, 8));

		// Then swap all 16-bit words													
		a = simde_mm_shufflelo_epi16(a, _MM_SHUFFLE(2, 3, 0, 1));
		a = simde_mm_shufflehi_epi16(a, _MM_SHUFFLE(2, 3, 0, 1));
		return a;
#endif
	}

	/// Reverse order of bytes in each 64-bit word											
	inline simde__m128i _mm_bswap_epi64(simde__m128i x) {
#ifdef __SSSE3__
		return simde_mm_shuffle_epi8(x, simde_mm_set_epi8(
			8, 9, 10, 11,
			12, 13, 14, 15,
			0, 1, 2, 3,
			4, 5, 6, 7
		));
#else
		// Swap bytes in each 16-bit word											
		simde__m128i a = simde_mm_or_si128(simde_mm_slli_epi16(x, 8), simde_mm_srli_epi16(x, 8));

		// Reverse all 16-bit words in 64-bit halves								
		a = simde_mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
		a = simde_mm_shufflehi_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
		return a;
#endif
	}

	/// Reverse order of all bytes in the 128-bit word										
	inline simde__m128i _mm_bswap_si128(simde__m128i x) {
#ifdef __SSSE3__
		return simde_mm_shuffle_epi8(x, simde_mm_set_epi8(
			0, 1, 2, 3,
			4, 5, 6, 7,
			8, 9, 10, 11,
			12, 13, 14, 15
		));
#else
		// Swap bytes in each 16-bit word											
		simde__m128i a = simde_mm_or_si128(simde_mm_slli_epi16(x, 8), simde_mm_srli_epi16(x, 8));

		// Reverse all 16-bit words in 64-bit halves								
		a = simde_mm_shufflelo_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));
		a = simde_mm_shufflehi_epi16(a, _MM_SHUFFLE(0, 1, 2, 3));

		// Reverse 64-bit halves														
		return simde_mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2));
#endif
	}

	/// Returns 0xFFFF where x <= y																
	inline simde__m128i _mm_cmple_epu16(simde__m128i x, simde__m128i y) {
		return simde_mm_cmpeq_epi16(simde_mm_subs_epu16(x, y), simde_mm_setzero_si128());
	}

	/// Returns 0xFF where x <= y																	
	inline simde__m128i _mm_cmple_epu8(simde__m128i x, simde__m128i y) {
		return simde_mm_cmpeq_epi8(simde_mm_min_epu8(x, y), x);
	}

	/// Returns 0xFFFF where x >= y																
	inline simde__m128i _mm_cmpge_epu16(simde__m128i x, simde__m128i y) {
		return _mm_cmple_epu16(y, x);
	}

	/// Returns 0xFF where x >= y																	
	inline simde__m128i _mm_cmpge_epu8(simde__m128i x, simde__m128i y) {
		return _mm_cmple_epu8(y, x);
	}

	/// Returns 0xFFFF where x > y																
	inline simde__m128i _mm_cmpgt_epu16(simde__m128i x, simde__m128i y) {
		return simde_mm_andnot_si128(simde_mm_cmpeq_epi16(x, y), _mm_cmple_epu16(y, x));
	}

	/// Returns 0xFF where x > y																	
	inline simde__m128i _mm_cmpgt_epu8(simde__m128i x, simde__m128i y) {
		return simde_mm_andnot_si128(
			simde_mm_cmpeq_epi8(x, y),
			simde_mm_cmpeq_epi8(simde_mm_max_epu8(x, y), x)
		);
	}

	/// Returns 0xFFFF where x < y																
	inline simde__m128i _mm_cmplt_epu16(simde__m128i x, simde__m128i y) {
		return _mm_cmpgt_epu16(y, x);
	}

	/// Returns 0xFF where x < y																	
	inline simde__m128i _mm_cmplt_epu8(simde__m128i x, simde__m128i y) {
		return _mm_cmpgt_epu8(y, x);
	}

	/// Divide 8 16-bit uints by 255																
	/// x := ((x + 1) + (x >> 8)) >> 8															
	inline simde__m128i _mm_div255_epu16(simde__m128i x) {
		return simde_mm_srli_epi16(simde_mm_adds_epu16(
			simde_mm_adds_epu16(x, simde_mm_set1_epi16(1)),
			simde_mm_srli_epi16(x, 8)
		), 8);
	}

	/// Find shift factor - this is actually much faster than							
	/// using __builtin_clz()																		
	inline simde__m128i _mm_divfast_epu8(simde__m128i x, uint8_t d) {
		uint8_t n
			= (d >= 128) ? 15
			: (d >= 64) ? 14
			: (d >= 32) ? 13
			: (d >= 16) ? 12
			: (d >= 8) ? 11
			: (d >= 4) ? 10
			: (d >= 2) ? 9
			: 8;

		// Set 8 words of "inverse sensitivity"										
		// Multiplying by this amount and right-shifting will give a			
		// very good approximation of the result										
		simde__m128i inv = simde_mm_set1_epi16((1 << n) / d + 1);

		// Unpack input into two 16-bit uints											
		simde__m128i lo = simde_mm_unpacklo_epi8(x, simde_mm_setzero_si128());
		simde__m128i hi = simde_mm_unpackhi_epi8(x, simde_mm_setzero_si128());

		// Multiply with the "inverse sensitivity" and divide						
		lo = simde_mm_srli_epi16(simde_mm_mullo_epi16(lo, inv), n);
		hi = simde_mm_srli_epi16(simde_mm_mullo_epi16(hi, inv), n);

		// Repack																				
		return simde_mm_packus_epi16(lo, hi);
	}

	/*#ifndef __SSE4_1__
	/// Returns x where x >= y, else y
	inline __m128i _mm_max_epu16(__m128i x, __m128i y) {
		return _mm_blendv_si128(x, y, _mm_cmple_epu16(x, y));
	}

	/// Returns x where x <= y, else y
	inline __m128i _mm_min_epu16(__m128i x, __m128i y) {
		return _mm_blendv_si128(y, x, _mm_cmple_epu16(x, y));
	}
	#endif*/

	/// Returns ~x, the bitwise complement of x												
	inline simde__m128i _mm_not_si128(simde__m128i x) {
		return simde_mm_xor_si128(x, simde_mm_cmpeq_epi32(simde_mm_setzero_si128(), simde_mm_setzero_si128()));
	}

	/// Returns an "alpha blend" of x scaled by y/255										
	///   x := x * (y / 255)																		
	/// Reorder: x := (x * y) / 255																
	inline simde__m128i _mm_scale_epu8(simde__m128i x, simde__m128i y) {
		// Unpack x and y into 16-bit uints												
		simde__m128i xlo = simde_mm_unpacklo_epi8(x, simde_mm_setzero_si128());
		simde__m128i ylo = simde_mm_unpacklo_epi8(y, simde_mm_setzero_si128());
		simde__m128i xhi = simde_mm_unpackhi_epi8(x, simde_mm_setzero_si128());
		simde__m128i yhi = simde_mm_unpackhi_epi8(y, simde_mm_setzero_si128());

		// Multiply x with y, keeping the low 16 bits								
		xlo = simde_mm_mullo_epi16(xlo, ylo);
		xhi = simde_mm_mullo_epi16(xhi, yhi);

		// Divide by 255																		
		xlo = _mm_div255_epu16(xlo);
		xhi = _mm_div255_epu16(xhi);

		// Repack the 16-bit uints to clamped 8-bit values							
		return simde_mm_packus_epi16(xlo, xhi);
	}

} // namespace Langulus::SIMD