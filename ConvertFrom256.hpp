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

	/// Convert __m256 to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	LANGULUS(ALWAYSINLINE) auto ConvertFrom256(const simde__m256& v) noexcept {
		//																						
		// Converting FROM float[8]													
		//																						
		if constexpr (CT::Same<TO, simde__m128d> && S <= 2) {
			//																					
			// Converting TO double[2]													
			//																					
			// float[2] -> double[2]													
			return simde_mm_cvtps_pd(simde_mm256_castps256_ps128(v));
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			//																					
			// Converting TO  pci8[8], pcu8[8], pci16[8], pcu16[8]			
			//						pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (CT::SignedInteger8<TT> && S <= 8) {
				// float[8] -> pci8[8]													
				auto
				vi32_16_8 = simde_mm256_cvtps_epi32(v);
				vi32_16_8 = simde_mm256_packs_epi32(vi32_16_8, vi32_16_8);
				vi32_16_8 = simde_mm256_packs_epi16(vi32_16_8, vi32_16_8);
				return simde_mm256_castsi256_si128(vi32_16_8);
			}
			else if constexpr (CT::UnsignedInteger8<TT> && S <= 8) {
				// float[8] -> pcu8[8]													
				auto
				vu32_16_8 = simde_mm256_cvtps_epi32(v);
				vu32_16_8 = simde_mm256_packus_epi32(vu32_16_8, vu32_16_8);
				vu32_16_8 = simde_mm256_packus_epi16(vu32_16_8, vu32_16_8);
				return simde_mm256_castsi256_si128(vu32_16_8);
			}
			else if constexpr (CT::SignedInteger16<TT> && S <= 8) {
				// float[8] -> pci16[8]													
				auto
				vi32_16 = simde_mm256_cvtps_epi32(v);
				vi32_16 = simde_mm256_packs_epi32(vi32_16, vi32_16);
				return simde_mm256_castsi256_si128(vi32_16);
			}
			else if constexpr (CT::UnsignedInteger16<TT> && S <= 8) {
				// float[8] -> pcu16[8]													
				auto
				vu32_16 = simde_mm256_cvtps_epi32(v);
				vu32_16 = simde_mm256_packus_epi32(vu32_16, vu32_16);
				return simde_mm256_castsi256_si128(vu32_16);
			}
			else if constexpr (CT::Same<TT, int32_t> && S <= 4) {
				// float[4] -> pci32[4]													
				return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
			}
			else if constexpr (CT::Same<TT, uint32_t> && S <= 4) {
				// float[4] -> pcu32[4]													
				return simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
			}
			else if constexpr (CT::Same<TT, int64_t> && S <= 2) {
				// float[2] -> pci64[2]													
				const auto vi32 = simde_mm256_cvtps_epi32(v);
				const auto vi32_128 = simde_mm256_castsi256_si128(vi32);
				const auto vi64 = simde_mm256_cvtepi32_epi64(vi32_128);
				return simde_mm256_castsi256_si128(vi64);
			}
			else if constexpr (CT::Same<TT, uint64_t> && S <= 2) {
				// float[2] -> pcu64[2]													
				const auto vu32 = simde_mm256_cvtps_epi32(v);
				const auto vu32_128 = simde_mm256_castsi256_si128(vu32);
				const auto vu64 = simde_mm256_cvtepi32_epi64(vu32_128);
				return simde_mm256_castsi256_si128(vu64);
			}
			else LANGULUS_ASSERT("Can't convert from __m256 to __m128i");
		}
		else if constexpr (CT::Same<TO, simde__m256i>) {
			//																					
			// Converting TO pci64[4], pcu64[4]										
			//																					
			if constexpr (CT::Same<TT, int64_t> && S <= 4) {
				// float[4] -> pci64[4]													
				const auto v32 = simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
				return simde_mm256_cvtepi32_epi64(v32);
			}
			else if constexpr (CT::Same<TT, uint64_t> && S <= 4) {
				// float[4] -> pcu64[4] 												
				const auto v32 = simde_mm_cvtps_epi32(simde_mm256_castps256_ps128(v));
				return simde_mm256_cvtepu32_epi64(v32);
			}
			else LANGULUS_ASSERT("Can't convert from __m256 to __m256i");
		}
		else if constexpr (CT::Same<TO, simde__m256d> && S <= 4) {
			//																					
			// Converting TO double[4]													
			//																					
			// float[4] -> double[4]													
			return simde_mm256_cvtps_pd(simde_mm256_castps256_ps128(v));
		}
		else LANGULUS_ASSERT("Can't convert from __m256 to unsupported");
	}

	/// Convert __m256d to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	LANGULUS(ALWAYSINLINE) auto ConvertFrom256d(const simde__m256d& v) noexcept {
		//																						
		// Converting FROM double[4]													
		//																						
		if constexpr (CT::Same<TO, simde__m128> && S <= 4) {
			//																					
			// Converting TO float[4]													
			//																					
			// double[4] -> float[4]													
			return simde_mm256_cvtpd_ps(v);
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			//																					
			// Converting TO	pci8[4], pcu8[4], pci16[4], pcu16[4]			
			//						pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (CT::SignedInteger8<TT> && S <= 4) {
				// double[4] -> pci8[4]													
				auto
				vi32_16_8 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
				vi32_16_8 = simde_mm_packs_epi32(vi32_16_8, vi32_16_8);
				vi32_16_8 = simde_mm_packs_epi16(vi32_16_8, vi32_16_8);
				return vi32_16_8;
			}
			else if constexpr (CT::UnsignedInteger8<TT> && S <= 4) {
				// double[4] -> pcu8[4]													
				auto
				vu32_16_8 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
				vu32_16_8 = simde_mm_packus_epi32(vu32_16_8, vu32_16_8);
				vu32_16_8 = simde_mm_packus_epi16(vu32_16_8, vu32_16_8);
				return vu32_16_8;
			}
			else if constexpr (CT::SignedInteger16<TT> && S <= 4) {
				// double[4] -> pci16[4]												
				auto
				vi32_16 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
				vi32_16 = simde_mm_packs_epi32(vi32_16, vi32_16);
				return vi32_16;
			}
			else if constexpr (CT::UnsignedInteger16<TT> && S <= 4) {
				// double[4] -> pcu16[4]												
				auto
				vu32_16 = simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
				vu32_16 = simde_mm_packus_epi32(vu32_16, vu32_16);
				return vu32_16;
			}
			else if constexpr (CT::Integer32<TT> && S <= 4) {
				// double[4] -> pci32[4] 												
				// double[4] -> pcu32[4] 												
				return simde_mm_cvtps_epi32(simde_mm256_cvtpd_ps(v));
			}
			else if constexpr (CT::Integer64<TT> && S <= 2) {
				// double[2] -> pci64[2]												
				// double[2] -> pcu64[2]												
				const auto v32 = simde_mm256_cvtpd_epi32(v);
				return simde_mm_unpacklo_epi32(v32, simde_mm_setzero_si128());
			}
			else LANGULUS_ASSERT("Can't convert from __m256d to __m128i");
		}
		else LANGULUS_ASSERT("Can't convert from __m256d to unsupported");
	}

	/// Convert __m256i to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	LANGULUS(ALWAYSINLINE) auto ConvertFrom256i(const simde__m256i& v) noexcept {
		//																						
		// Converting FROM	pci8[32], pcu8[32], pci16[16], pcu16[16]		
		//							pci32[8], pcu32[8], pci64[4], pcu64[4]			
		//																						
		if constexpr (CT::Same<TO, simde__m128>) {
			//																					
			// Converting TO float[4]													
			//																					
			if constexpr (CT::Same<int8_t, FT> && S <= 4) {
				// pci8[4] -> float[4]													
				const auto v32 = simde_mm_cvtepi8_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_ps(v32);
			}
			else if constexpr (CT::Same<uint8_t, FT> && S <= 4) {
				// pci8[4] -> float[4]													
				const auto v32 = simde_mm_cvtepu8_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_ps(v32);
			}
			else if constexpr (CT::Same<int16_t, FT> && S <= 4) {
				// pci16[4] -> float[4]													
				const auto v32 = simde_mm_cvtepi16_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_ps(v32);
			}
			else if constexpr (CT::Same<uint16_t, FT> && S <= 4) {
				// pci16[4] -> float[4]													
				const auto v32 = simde_mm_cvtepu16_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_ps(v32);
			}
			else if constexpr (CT::Integer32<FT> && S <= 4) {
				// pci32[4] -> float[4]													
				// pcu32[4] -> float[4]													
				return simde_mm_cvtepi32_ps(simde_mm256_castsi256_si128(v));
			}
			else if constexpr (CT::UnsignedInteger64<FT> && S <= 4) {
				// pcu64[4] -> float[4]													
				//TODO generalize this when 512 stuff is added to SIMDe		
				#if LANGULUS_SIMD(AVX512DQ) && LANGULUS_SIMD(AVX512VL)
					return simde_mm256_cvtepu64_ps(v);
				#else
					auto m1 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
					auto m2 = uint64_to_double_full(simde_mm256_extracti128_si256(v, 1));
					return simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2));
				#endif
			}
			else if constexpr (CT::SignedInteger64<FT> && S <= 4) {
				// pci64[4] -> float[4]													
				//TODO generalize this when 512 stuff is added to SIMDe		
				#if LANGULUS_SIMD(AVX512DQ) && LANGULUS_SIMD(AVX512VL)
					return simde_mm256_cvtepi64_ps(v);
				#else
					auto m1 = int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
					auto m2 = int64_to_double_full(simde_mm256_extracti128_si256(v, 1));
					return simde_mm_movelh_ps(simde_mm_cvtpd_ps(m1), simde_mm_cvtpd_ps(m2));
				#endif
			}
			else LANGULUS_ASSERT("Can't convert from __m256i to __m128");
		}
		else if constexpr (CT::Same<TO, simde__m128d>) {
			//																					
			// Converting TO double[2]													
			//																					
			if constexpr (CT::Same<FT, int8_t> && S <= 2) {
				// pci8[2] -> double[2]													
				const auto v32 = simde_mm_cvtepi8_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_pd(v32);
			}
			else if constexpr (CT::Same<FT, uint8_t> && S <= 2) {
				// pcu8[2] -> double[2]													
				const auto v32 = simde_mm_cvtepu8_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_pd(v32);
			}
			else if constexpr (CT::Same<FT, int16_t> && S <= 2) {
				// pci16[2] -> double[2]												
				const auto v32 = simde_mm_cvtepi16_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_pd(v32);
			}
			else if constexpr (CT::Same<FT, uint16_t> && S <= 2) {
				// pcu16[2] -> double[2]												
				const auto v32 = simde_mm_cvtepu16_epi32(simde_mm256_castsi256_si128(v));
				return simde_mm_cvtepi32_pd(v32);
			}
			else if constexpr (CT::Integer32<FT> && S <= 2) {
				// pci32[2] -> double[2]												
				// pcu32[2] -> double[2]												
				return simde_mm_cvtepi32_pd(simde_mm256_castsi256_si128(v));
			}
			else if constexpr (CT::UnsignedInteger64<FT> && S <= 2) {
				// pcu64[2] -> double[2]												
				//TODO generalize this when 512 stuff is added to SIMDe		
				#if LANGULUS_SIMD(AVX512DQ) && LANGULUS_SIMD(AVX512VL)
					return simde_mm_cvtepu64_pd(simde_mm256_extracti128_si256(v, 0));
				#else
					return uint64_to_double_full(simde_mm256_extracti128_si256(v, 0));
				#endif
			}
			else if constexpr (CT::SignedInteger64<FT> && S <= 2) {
				// pci64[2] -> double[2]												
				//TODO generalize this when 512 stuff is added to SIMDe		
				#if LANGULUS_SIMD(AVX512DQ) && LANGULUS_SIMD(AVX512VL)
					return simde_mm_cvtepi64_pd(simde_mm256_extracti128_si256(v, 0));
				#else
					return int64_to_double_full(simde_mm256_extracti128_si256(v, 0));
				#endif
			}
			else LANGULUS_ASSERT("Can't convert from __m256i to __m128d");
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			if constexpr (CT::Same<FT, int8_t>) {
				//																				
				// Converting TO	pci8[16], pcu8[16], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 16) {
					// pci8[16] -> pcu8[16]												
					// pci8[16] -> pci8[16]												
					return simde_mm256_castsi256_si128(v);
				}
				else if constexpr (CT::Integer16<TT> && S <= 8) {
					// pci8[8] -> pci16[8]												
					// pci8[8] -> pcu16[8]												
					return simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
				}
				else if constexpr (CT::Integer32<TT> && S <= 4) {
					// pci8[4] -> pci32[4]												
					// pci8[4] -> pcu32[4]												
					auto
					v16_32 = simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
					v16_32 = simde_mm_unpacklo_epi16(v16_32, simde_mm_setzero_si128());
					return v16_32;
				}
				else if constexpr (CT::Integer64<TT> && S <= 2) {
					// pci8[2] -> pci64[2]												
					// pci8[2] -> pcu64[2]												
					auto
					v16_32 = simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
					v16_32 = simde_mm_unpacklo_epi16(v16_32, simde_mm_setzero_si128());
					return simde_mm_cvtepi32_epi64(v16_32);
				}
				else LANGULUS_ASSERT("Can't convert from pci8 to unsupported TT");
			}
			else if constexpr (CT::Same<FT, uint8_t>) {
				//																				
				// Converting TO	pci8[16], pcu8[16], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 16) {
					// pcu8[16] -> pcu8[16]												
					// pcu8[16] -> pci8[16]												
					return simde_mm256_castsi256_si128(v);
				}
				else if constexpr (CT::Integer16<TT> && S <= 8) {
					// pcu8[8] -> pci16[8]												
					// pcu8[8] -> pcu16[8]												
					return simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
				}
				else if constexpr (CT::Integer32<TT> && S <= 4) {
					// pcu8[4] -> pci32[4]												
					// pcu8[4] -> pcu32[4]												
					auto
					v16_32 = simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
					v16_32 = simde_mm_unpacklo_epi16(v16_32, _mm_setzero_si128());
					return v16_32;
				}
				else if constexpr (CT::Integer64<TT> && S <= 2) {
					// pcu8[2] -> pci64[2]												
					// pcu8[2] -> pcu64[2]												
					auto
					v16_32 = simde_mm_unpacklo_epi8(simde_mm256_castsi256_si128(v), simde_mm_setzero_si128());
					v16_32 = simde_mm_unpacklo_epi16(v16_32, simde_mm_setzero_si128());
					return simde_mm_cvtepi32_epi64(v16_32);
				}
				else LANGULUS_ASSERT("Can't convert from pcu8 to unsupported TT");
			}
			else if constexpr (CT::Same<FT, int16_t>) {
				//																				
				// Converting TO	pci8[8], pcu8[8], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::SignedInteger8<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pci8[8]");
				}
				else if constexpr (CT::UnsignedInteger8<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pcu8[8]");
				}
				else if constexpr (CT::UnsignedInteger16<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[8] to pcu16[8]");
				}
				else if constexpr (CT::Same<TT, int32_t> && S <= 4) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[4] to pci32[4]");
				}
				else if constexpr (CT::Same<TT, uint32_t> && S <= 4) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[4] to pcu32[4]");
				}
				else if constexpr (CT::Same<TT, int64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[2] to pci64[2]");
				}
				else if constexpr (CT::Same<TT, uint64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pci16[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci16 to unsupported TT");
			}
			else if constexpr (CT::Same<FT, uint16_t>) {
				//																				
				// Converting TO	pci8[8], pcu8[8], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::SignedInteger8<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pci8[8]");
				}
				else if constexpr (CT::UnsignedInteger8<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pcu8[8]");
				}
				else if constexpr (CT::SignedInteger16<TT> && S <= 8) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[8] to pci16[8]");
				}
				else if constexpr (CT::Same<TT, int32_t> && S <= 4) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[4] to pci32[4]");
				}
				else if constexpr (CT::Same<TT, uint32_t> && S <= 4) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[4] to pcu32[4]");
				}
				else if constexpr (CT::Same<TT, int64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[2] to pci64[2]");
				}
				else if constexpr (CT::Same<TT, uint64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]												
					LANGULUS_ASSERT("Can't convert from pcu16[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pcu16 to unsupported TT");
			}
			else if constexpr (CT::Integer32<FT>) {
				//																				
				// Converting TO	pci8[4], pcu8[4], pci16[4], pcu16[4]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::SignedInteger8<TT> && S <= 4) {
					// pci32[4] -> pci8[4]												
					auto
					v16_8 = simde_mm256_castsi256_si128(v);
					v16_8 = simde_mm_packs_epi32(v16_8, v16_8);
					v16_8 = simde_mm_packs_epi16(v16_8, v16_8);
					return v16_8;
				}
				else if constexpr (CT::UnsignedInteger8<TT> && S <= 4) {
					// pci32[4] -> pcu8[4]												
					auto
					v16_8 = simde_mm256_castsi256_si128(v);
					v16_8 = simde_mm_packus_epi32(v16_8, v16_8);
					v16_8 = simde_mm_packus_epi16(v16_8, v16_8);
					return v16_8;
				}
				else if constexpr (CT::SignedInteger16<TT> && S <= 4) {
					// pci32[4] -> pci16[4]												
					auto
					v16_8 = simde_mm256_castsi256_si128(v);
					v16_8 = simde_mm_packs_epi32(v16_8, v16_8);
					return v16_8;
				}
				else if constexpr (CT::UnsignedInteger16<TT> && S <= 4) {
					// pci32[4] -> pcu16[4]												
					auto
					v16_8 = simde_mm256_castsi256_si128(v);
					v16_8 = simde_mm_packus_epi32(v16_8, v16_8);
					return v16_8;
				}
				else if constexpr (CT::Integer32<TT> && S <= 4) {
					// pci32[4] -> pci32[4]												
					// pci32[4] -> pcu32[4]												
					return simde_mm256_castsi256_si128(v);
				}
				else if constexpr (CT::Same<TT, int64_t> && S <= 2) {
					// pci32[2] -> pci64[2]												
					const auto v64 = simde_mm256_cvtepi32_epi64(simde_mm256_castsi256_si128(v));
					return simde_mm256_castsi256_si128(v64);
				}
				else if constexpr (CT::Same<TT, uint64_t> && S <= 2) {
					// pci32[2] -> pcu64[2]												
					const auto v64 = simde_mm256_cvtepu32_epi64(simde_mm256_castsi256_si128(v));
					return simde_mm256_castsi256_si128(v64);
				}
				else LANGULUS_ASSERT("Can't convert from pci32 to unsupported TT");
			}
			else if constexpr (CT::Same<FT, int64_t>) {
				//																				
				// Converting TO	pci8[2], pcu8[2], pci16[2], pcu16[2]		
				//						pci32[2], pcu32[2], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::SignedInteger8<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci8[2]");
				}
				else if constexpr (CT::UnsignedInteger8<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu8[2]");
				}
				else if constexpr (CT::SignedInteger16<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci16[2]");
				}
				else if constexpr (CT::UnsignedInteger16<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu16[2]");
				}
				else if constexpr (CT::Same<TT, int32_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pci32[2]");
				}
				else if constexpr (CT::Same<TT, uint32_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu32[2]");
				}
				else if constexpr (CT::Same<TT, uint64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pci64[2] to pcu64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pci64 to unsupported TT");
			}
			else if constexpr (CT::Same<FT, uint64_t>) {
				//																				
				// Converting TO	pci8[2], pcu8[2], pci16[2], pcu16[2]		
				//						pci32[2], pcu32[2], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::SignedInteger8<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci8[2]");
				}
				else if constexpr (CT::UnsignedInteger8<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu8[2]");
				}
				else if constexpr (CT::SignedInteger16<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci16[2]");
				}
				else if constexpr (CT::UnsignedInteger16<TT> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu16[2]");
				}
				else if constexpr (CT::Same<TT, int32_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci32[2]");
				}
				else if constexpr (CT::Same<TT, uint32_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pcu32[2]");
				}
				else if constexpr (CT::Same<TT, int64_t> && S <= 2) {
					// pci8[16] -> pcu8[16]											
					LANGULUS_ASSERT("Can't convert from pcu64[2] to pci64[2]");
				}
				else LANGULUS_ASSERT("Can't convert from pcu64 to unsupported TT");
			}
			else LANGULUS_ASSERT("Can't convert from __m256i to __m128d");
		}
		else LANGULUS_ASSERT("Can't convert from __m256i to unsupported");
	}

} // namespace Langulus::SIMD