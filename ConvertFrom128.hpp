///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Load.hpp"

namespace Langulus::SIMD
{

	/// Convert __m128 to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - register to convert to												
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom128(const simde__m128& v) noexcept {
		//																						
		// Converting FROM float[4]													
		//																						
		if constexpr (Same<TO, simde__m128d> && S <= 2) {
			//																					
			// Converting TO double[2]													
			//																					
			// float[2] -> double[2]													
			return simde_mm_cvtps_pd(v);
		}
		else if constexpr (Same<TO, simde__m128i>) {
			//																					
			// Converting TO pci8[4], pcu8[4], pci16[4], pcu16[4]				
			//					  pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (SignedInteger8<TT> && S <= 4) {
				// float[4] -> pci8[4]													
				auto 
				vi32_16_8 = simde_mm_cvtps_epi32(v);
				vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
				vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
				return vi32_16_8;
			}
			else if constexpr (UnsignedInteger8<TT> && S <= 4) {
				// float[4] -> pcu8[4]													
				auto 
				vi32_16_8 = simde_mm_cvtps_epi32(v);
				vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
				vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
				return vi32_16_8;
			}
			else if constexpr (SignedInteger16<TT> && S <= 4) {
				// float[4] -> pci16[4]													
				auto 
				vi32_16 = simde_mm_cvtps_epi32(v);
				vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
				return vi32_16;
			}
			else if constexpr (UnsignedInteger16<TT> && S <= 4) {
				// float[4] -> pcu16[4]													
				auto 
				vi32_16 = simde_mm_cvtps_epi32(v);
				vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
				return vi32_16;
			}
			else if constexpr (SignedInteger32<TT> && S <= 4) {
				// float[4] -> pci32[4]													
				return simde_mm_cvtps_epi32(v);
			}
			else if constexpr (UnsignedInteger32<TT> && S <= 4) {
				// float[4] -> pcu32[4]													
				return _mm_cvtps_epu32(v);
			}
			else if constexpr (SignedInteger64<TT> && S <= 2) {
				// float[2] -> pci64[2]													
				return _mm_cvtps_epi64(v);
			}
			else if constexpr (UnsignedInteger64<TT> && S <= 2) {
				// float[2] -> pcu64[2]													
				return _mm_cvtps_epu64(v);
			}
			else LANGULUS_ASSERT("Can't convert from __m128 to __m128i");
		}
		else if constexpr (Same<TO, simde__m256i>) {
			//																					
			// Converting TO pci64[4], pcu64[4]										
			//																					
			if constexpr (SignedInteger64<TT> && S <= 4) {
				// float[4] -> pci64[4]													
				return _mm256_cvtps_epi64(v);
			}
			else if constexpr (UnsignedInteger64<TT> && S <= 4) {
				// float[4] -> pcu64[4]													
				return _mm256_cvtps_epu64(v);
			}
			else LANGULUS_ASSERT("Can't convert from __m128 to __m256i");
		}
		else if constexpr (Same<TO, simde__m256d> && S <= 4) {
			//																					
			// Converting TO double[4]													
			//																					
			// float[4] -> double[4]													
			return simde_mm256_cvtps_pd(v);
		}
		else LANGULUS_ASSERT("Can't convert from __m128 to unsupported");
	}

	/// Convert __m128d to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - register to convert to												
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom128d(const simde__m128d& v) noexcept {
		//																						
		// Converting FROM double[2]													
		//																						
		if constexpr (Same<TO, simde__m128> && S <= 2) {
			//																					
			// Converting TO float[2]													
			//																					
			// double[2] -> float[2]													
			return simde_mm_cvtpd_ps(v);
		}
		else if constexpr (Same<TO, simde__m128i>) {
			//																					
			// Converting TO pci8[2], pcu8[2], pci16[2], pcu16[2]				
			//					  pci32[2], pcu32[2], pci64[2], pcu64[2]			
			//																					
			if constexpr (SignedInteger8<TT> && S <= 2) {
				// double[2] -> pci8[2]													
				auto 
				vi32_16_8 = simde_mm_cvtpd_epi32(v);
				vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, simde_mm_setzero_si128());
				vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, simde_mm_setzero_si128());
				return vi32_16_8;
			}
			else if constexpr (UnsignedInteger8<TT> && S <= 2) {
				// double[2] -> pcu8[2]													
				auto 
				vi32_16_8 = simde_mm_cvtpd_epi32(v);
				vi32_16_8 = simde_mm_packus_epi32(vi32_16_8, simde_mm_setzero_si128());
				vi32_16_8 = simde_mm_packus_epi16(vi32_16_8, simde_mm_setzero_si128());
				return vi32_16_8;
			}
			else if constexpr (SignedInteger16<TT> && S <= 2) {
				// double[2] -> pci16[2]												
				auto 
				vi32_16 = simde_mm_cvtpd_epi32(v);
				vi32_16 = simde_mm_packs_epi32(vi32_16, simde_mm_setzero_si128());
				return vi32_16;
			}
			else if constexpr (UnsignedInteger16<TT> && S <= 2) {
				// double[2] -> pcu16[2]												
				auto 
				vi32_16 = simde_mm_cvtpd_epi32(v);
				vi32_16 = simde_mm_packus_epi32(vi32_16, simde_mm_setzero_si128());
				return vi32_16;
			}
			else if constexpr (Integer32<TT> && S <= 2) {
				// double[2] -> pci32[2] or pcu32[2]								
				return simde_mm_cvtpd_pi32(v);
			}
			else if constexpr (SignedInteger64<TT> && S <= 2) {
				// double[2] -> pci64[2] or pcu64[2]								
				return _mm_cvtpd_epi64(v);
			}
			else if constexpr (UnsignedInteger64<TT> && S <= 2) {
				// double[2] -> pci64[2] or pcu64[2]								
				return _mm_cvtpd_epu64(v);
			}
			else LANGULUS_ASSERT("Can't convert from __m128d to __m128i");
		}
		else LANGULUS_ASSERT("Can't convert from __m128d to unsupported");
	}

	/// Convert __m128i to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - register to convert to												
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom128i(const simde__m128i& v) noexcept {
		//																						
		// Converting FROM pci8[16], pcu8[16], pci16[8], pcu16[8]			
		//						 pci32[4], pcu32[4], pci64[2], pcu64[2]			
		//																						
		if constexpr (Same<TO, simde__m128>) {
			//																					
			// Converting TO float[4] or float[2]									
			//																					
			if constexpr (SignedInteger8<FT> && S <= 4) {
				// pci8[4] -> float[4]													
				auto
				vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
				vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_ps(vi16_32);											
			}
			else if constexpr (UnsignedInteger8<FT> && S <= 4) {
				// pcu8[4] -> float[4]													
				auto
				vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
				vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
				return _mm_cvtepu32_ps(vi16_32);
			}
			else if constexpr (SignedInteger16<FT> && S <= 4) {
				// pci16[4] -> float[4]													
				auto 
				vi32 = simde_mm_unpacklo_epi16(vi32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_ps(vi32);											
			}
			else if constexpr (UnsignedInteger16<FT> && S <= 4) {
				// pcu16[4] -> float[4]													
				auto 
				cvt = simde_mm_unpacklo_epi16(cvt, simde_mm_setzero_si128());
				return _mm_cvtepu32_ps(cvt);
			}
			else if constexpr (SignedInteger32<FT> && S <= 4) {
				// pci32[4] -> float[4]													
				return simde_mm_cvtepi32_ps(v);
			}
			else if constexpr (UnsignedInteger32<FT> && S <= 4) {
				// pcu32[4] -> float[4]													
				return simde_mm256_cvtpd_ps(simde_mm256_cvtepu32_pd(v));
			}
			else if constexpr (SignedInteger64<FT> && S <= 2) {
				// pci64[2] -> float[2]													
				return simde_mm_cvtepi64_ps(v);
			}
			else if constexpr (UnsignedInteger64<FT> && S <= 2) {
				// pcu64[2] -> float[2]													
				return simde_mm_cvtepu64_ps(v);
			}
			else LANGULUS_ASSERT("Can't convert from __m128i to __m128");
		}
		else if constexpr (Same<TO, simde__m128d>) {
			//																					
			// Converting TO double[2]													
			//																					
			if constexpr (SignedInteger8<FT> && S <= 2) {
				// pci8[2] -> double[2]													
				auto
				vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
				vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_pd(vi16_32);											
			}
			else if constexpr (UnsignedInteger8<FT> && S <= 2) {
				// pcu8[2] -> double[2]													
				auto
				vi16_32 = simde_mm_unpacklo_epi8(v, simde_mm_setzero_si128());
				vi16_32 = simde_mm_unpacklo_epi16(vi16_32, simde_mm_setzero_si128());
				return _mm_cvtepu32_pd(vi16_32);											
			}
			else if constexpr (SignedInteger16<FT> && S <= 2) {
				// pci16[2] -> double[2]												
				auto 
				vi32 = simde_mm_unpacklo_epi16(vi32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_pd(vi32);											
			}
			else if constexpr (UnsignedInteger16<FT> && S <= 2) {
				// pcu16[2] -> double[2]												
				auto 
				vi32 = simde_mm_unpacklo_epi16(vi32, simde_mm_setzero_si128());
				return _mm_cvtepu32_pd(vi32);											
			}
			else if constexpr (SignedInteger32<FT> && S <= 2) {
				// pci32[2] -> double[2]												
				return simde_mm_cvtepi32_pd(v);
			}
			else if constexpr (UnsignedInteger32<FT> && S <= 2) {
				// pcu32[2] -> double[2]												
				return _mm_cvtepu32_pd(v);
			}
			else if constexpr (SignedInteger64<FT> && S <= 2) {
				// pci64[2] -> double[2]												
				return _mm_cvtepi64_pd(v);
			}
			else if constexpr (UnsignedInteger64<FT> && S <= 2) {
				// pcu64[2] -> double[2]												
				return _mm_cvtepu64_pd(v);
			}
			else LANGULUS_ASSERT("Can't convert from __m128i to __m128d");
		}
		else if constexpr (Same<TO, simde__m128i>) {
			//																					
			// Converting TO pci8[16], pcu8[16], pci16[8], pcu16[8]			
			//					  pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (SignedInteger8<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 16) {
					// pci8[16] -> pci8[16]												
					LANGULUS_ASSERT("Can't convert from pci8[16] to pci8[16]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 16) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci8[16] to pcu8[16]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 8) {
					// pci8[8] -> pci16[8]												
					LANGULUS_ASSERT("Can't convert from pci8[8] to pci16[8]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 8) {
					// pci8[8] -> pci16[8]												
					LANGULUS_ASSERT("Can't convert from pci8[8] to pcu16[8]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pci8[4] -> pci32[4]												
					LANGULUS_ASSERT("Can't convert from pci8[4] to pci32[4]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pci8[4] -> pcu32[4]												
					LANGULUS_ASSERT("Can't convert from pci8[4] to pcu32[4]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pci8[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pci8[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pci8[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pci8[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci8 to unsupported TT");
			}
			else if constexpr (UnsignedInteger8<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 16) {
					// pcu8[16] -> pci8[16]												
					LANGULUS_ASSERT("Can't convert from pcu8[16] to pci8[16]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 16) {
					// pcu8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu8[16] to pci8[16]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 8) {
					// pcu8[8] -> pci16[8]												
					LANGULUS_ASSERT("Can't convert from pcu8[8] to pci16[8]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 8) {
					// pcu8[8] -> pcu16[8]												
					LANGULUS_ASSERT("Can't convert from pcu8[8] to pcu16[8]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pcu8[4] -> pci32[4]												
					LANGULUS_ASSERT("Can't convert from pcu8[4] to pci32[4]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pcu8[4] -> pcu32[4]												
					LANGULUS_ASSERT("Can't convert from pcu8[4] to pcu32[4]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pcu8[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pcu8[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pcu8[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pcu8[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pcu8 to unsupported TT");
			}
			else if constexpr (SignedInteger16<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 8) {
					// pci16[8] -> pci8[8]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pci8[8]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 8) {
					// pci16[8] -> pcu8[8]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pcu8[8]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 8) {
					// pci16[8] -> pci16[8]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pci16[8]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 8) {
					// pci16[8] -> pcu16[8]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pcu16[8]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pci16[4] -> pci32[4]												
					LANGULUS_ASSERT("Can't convert from pci16[4] to pci32[4]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pci16[4] -> pcu32[4]												
					LANGULUS_ASSERT("Can't convert from pci16[4] to pcu32[4]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pci16[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pci16[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pci16[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pci16[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci16 to unsupported TT");
			}
			else if constexpr (UnsignedInteger16<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 8) {
					// pcu16[8] -> pci8[8]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pci8[8]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 8) {
					// pcu16[8] -> pcu8[8]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pcu8[8]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 8) {
					// pcu16[8] -> pci16[8]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pci16[8]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 8) {
					// pcu16[8] -> pcu16[8]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pcu16[8]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pcu16[4] -> pci32[4]												
					LANGULUS_ASSERT("Can't convert from pcu16[4] to pci32[4]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pcu16[4] -> pcu32[4]												
					LANGULUS_ASSERT("Can't convert from pcu16[4] to pcu32[4]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pcu16[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pcu16[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pcu16[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pcu16[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pcu16 to unsupported TT");
			}
			else if constexpr (SignedInteger32<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 4) {
					// pci32[4] -> pci8[4]												
					return simde_mm_packs_epi16(v, simde_mm_setzero_si128());
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 4) {
					// pci32[4] -> pcu8[4]												
					return simde_mm_packus_epi16(v, simde_mm_setzero_si128());
				}
				else if constexpr (SignedInteger16<TT> && S <= 4) {
					// pci32[4] -> pci16[4]												
					return simde_mm_packs_epi32(v, simde_mm_setzero_si128());
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 4) {
					// pci32[4] -> pcu16[4]												
					return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pci32[4] -> pci32[4]												
					return v;
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pci32[4] -> pcu32[4]												
					return v;
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci32[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci32[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci32 to unsupported TT");
			}
			else if constexpr (UnsignedInteger32<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 4) {
					// pcu32[4] -> pci8[4]												
					return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 4) {
					// pcu32[4] -> pcu8[4]												
					return simde_mm_packus_epi16(simde_mm_packus_epi32(v, simde_mm_setzero_si128()), simde_mm_setzero_si128());
				}
				else if constexpr (SignedInteger16<TT> && S <= 4) {
					// pcu32[4] -> pci16[4]												
					return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 4) {
					// pcu32[4] -> pcu16[4]												
					return simde_mm_packus_epi32(v, simde_mm_setzero_si128());
				}
				else if constexpr (SignedInteger32<TT> && S <= 4) {
					// pcu32[4] -> pci32[4]												
					auto lo = _mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(v));
					auto up = _mm_halfflip(_mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(_mm_halfflip(v))));
					return simde_mm_add_epi32(lo, up);
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 4) {
					// pcu32[4] -> pci32[4]												
					auto lo = _mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(v));
					auto up = _mm_halfflip(_mm_cvtepi64_epi32(simde_mm_cvtepu32_epi64(_mm_halfflip(v))));
					return simde_mm_add_epi32(lo, up);
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pcu32[2] -> pci64[2]												
					return simde_mm_cvtepu32_epi64(v);
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pcu32[2] -> pcu64[2]												
					return simde_mm_cvtepu32_epi64(v);
				}
				else LANGULUS_ASSERT("Can't convert from pcu32 to unsupported TT");
			}
			else if constexpr (SignedInteger64<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 2) {
					// pci64[2] -> pci8[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci8[2]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 2) {
					// pci64[2] -> pcu8[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu8[2]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 2) {
					// pci64[2] -> pci16[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci16[2]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 2) {
					// pci64[2] -> pcu16[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu16[2]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 2) {
					// pci64[2] -> pci32[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci32[2]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 2) {
					// pci64[2] -> pcu32[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu32[2]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pci64[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pci64[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci64 to unsupported TT");
			}
			else if constexpr (UnsignedInteger64<FT>) {
				if constexpr (SignedInteger8<TT> && S <= 2) {
					// pcu64[2] -> pci8[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci8[2]");
				}
				else if constexpr (UnsignedInteger8<TT> && S <= 2) {
					// pcu64[2] -> pcu8[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu8[2]");
				}
				else if constexpr (SignedInteger16<TT> && S <= 2) {
					// pcu64[2] -> pci16[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci16[2]");
				}
				else if constexpr (UnsignedInteger16<TT> && S <= 2) {
					// pcu64[2] -> pcu16[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu16[2]");
				}
				else if constexpr (SignedInteger32<TT> && S <= 2) {
					// pcu64[2] -> pci32[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci32[2]");
				}
				else if constexpr (UnsignedInteger32<TT> && S <= 2) {
					// pcu64[2] -> pcu32[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu32[2]");
				}
				else if constexpr (SignedInteger64<TT> && S <= 2) {
					// pcu64[2] -> pci64[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci64[2]");
				}
				else if constexpr (UnsignedInteger64<TT> && S <= 2) {
					// pcu64[2] -> pcu64[2]												
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pcu64 to unsupported TT");
			}
			else LANGULUS_ASSERT("Can't convert from __m128i to __m128i");
		}
		else if constexpr (Same<TO, __m128d>) {
			//																					
			// Converting TO double[2]													
			//																					
			LANGULUS_ASSERT("Can't convert from __m128i to __m128d");
		}
		else LANGULUS_ASSERT("Can't convert from __m128i to unsupported");
	}

} // namespace Langulus::TSIMDe