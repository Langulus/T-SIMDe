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

	/// Convert __m512 to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom512(const simde__m512& v) noexcept {
		//																						
		// Converting FROM float[16]													
		//																						
		if constexpr (CT::Same<TO, simde__m128> && S <= 4) {
			//																					
			// Converting TO float[4]													
			//																					
			// float[4] -> float[4]														
			return simde_mm512_castps512_ps128(v);
		}
		else if constexpr (CT::Same<TO, simde__m128d> && S <= 2) {
			//																					
			// Converting TO double[2]													
			//																					
			// float[2] -> double[2]													
			return simde_mm_cvtps_pd(simde_mm512_castps512_ps128(v));
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			//																					
			// Converting TO  pci8[16], pcu8[16], pci16[8], pcu16[8]			
			//						pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (CT::SignedInteger8<TT> && S <= 16) {
				// float[16] -> pci8[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				v2 = simde_mm256_packs_epi32(v2, v2);
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packs_epi32(v1, v1);
				v1 = simde_mm256_packs_epi16(v1, v2);
				return simde_mm256_castsi256_si128(v1);
			}
			else if constexpr (CT::UnsignedInteger8<TT> && S <= 16) {
				// float[16] -> pcu8[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				v2 = simde_mm256_packus_epi32(v2, v2);
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packus_epi32(v1, v1);
				v1 = simde_mm256_packus_epi16(v1, v2);
				return simde_mm256_castsi256_si128(v1);
			}
			else if constexpr (CT::SignedInteger16<TT> && S <= 8) {
				// float[8] -> pci16[8]													
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packs_epi32(v1, v2);
				return simde_mm256_castsi256_si128(v1);
			}
			else if constexpr (CT::UnsignedInteger16<TT> && S <= 8) {
				// float[8] -> pcu16[8]													
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packus_epi32(v1, v2);
				return simde_mm256_castsi256_si128(v1);
			}
			else if constexpr (CT::Integer32<TT> && S <= 4) {
				// float[4] -> pci32[4]													
				// float[4] -> pcu32[4]													
				return simde_mm_cvtps_epi32(simde_mm512_castps512_ps128(v));
			}
			else if constexpr (CT::Integer64<TT> && S <= 2) {
				// float[2] -> pci64[2]													
				// float[2] -> pcu64[2]													
				const auto i32 = simde_mm_cvtps_epi32(simde_mm512_castps512_ps128(v));
				return simde_mm_cvtepi32_epi64(i32);
			}
			else LANGULUS_ASSERT("Can't convert from __m512 to __m128i");
		}
		else if constexpr (CT::Same<TO, simde__m256> && S <= 8) {
			//																					
			// Converting TO float[8]													
			//																					
			// float[8] -> float[8]														
			return simde_mm512_castps512_ps256(v);
		}
		else if constexpr (CT::Same<TO, simde__m256d> && S <= 4) {
			//																					
			// Converting TO double[4]													
			//																					
			// float[4] -> double[4]													
			return simde_mm256_cvtps_pd(simde_mm512_castps512_ps128(v));
		}
		else if constexpr (CT::Same<TO, simde__m256i>) {
			//																					
			// Converting TO  pci8[16], pcu8[16], pci16[16], pcu16[16]		
			//						pci32[8], pcu32[8], pci64[4], pcu64[4]			
			//																					
			if constexpr (CT::SignedInteger8<TT> && S <= 16) {
				// float[16] -> pci8[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				v2 = simde_mm256_packs_epi32(v2, v2);
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packs_epi32(v1, v1);
				v1 = simde_mm256_packs_epi16(v1, v2);
				return v1;
			}
			else if constexpr (CT::UnsignedInteger8<TT> && S <= 16) {
				// float[16] -> pcu8[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				v2 = simde_mm256_packus_epi32(v2, v2);
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packus_epi32(v1, v1);
				v1 = simde_mm256_packus_epi16(v1, v2);
				return v1;
			}
			else if constexpr (CT::SignedInteger16<TT> && S <= 16) {
				// float[16] -> pci16[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packs_epi32(v1, v2);
				return v1;
			}
			else if constexpr (CT::UnsignedInteger16<TT> && S <= 16) {
				// float[16] -> pcu16[16]												
				auto
				v2 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(_mm_halfflip(v)));
				auto
				v1 = simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
				v1 = simde_mm256_packus_epi32(v1, v2);
				return v1;
			}
			else if constexpr (CT::Integer32<TT> && S <= 8) {
				// float[8] -> pci32[8]													
				// float[8] -> pcu32[8]													
				return simde_mm256_cvtps_epi32(simde_mm512_castps512_ps256(v));
			}
			else if constexpr (CT::Integer64<TT> && S <= 4) {
				// float[4] -> pci64[4]													
				// float[4] -> pcu64[4]													
				const auto i32 = simde_mm_cvtps_epi32(simde_mm512_castps512_ps128(v));
				return simde_mm256_cvtepi32_epi64(i32);
			}
			else LANGULUS_ASSERT("Can't convert from __m512 to __m256i");
		}
		else LANGULUS_ASSERT("Can't convert from __m512 to unsupported");
	}

	/// Convert __m512d to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom512d(const simde__m512d& v) noexcept {
		//																						
		// Converting FROM double[8]													
		//																						
		if constexpr (CT::Same<TO, simde__m128> && S <= 4) {
			//																					
			// Converting TO float[4]													
			//																					
			// double[4] -> float[4]													
			return simde_mm256_cvtpd_ps(simde_mm512_castpd512_pd256(v));
		}
		else if constexpr (CT::Same<TO, simde__m128d> && S <= 2) {
			//																					
			// Converting TO double[2]													
			//																					
			// double[2] -> double[2]													
			return simde_mm512_castpd512_pd128(v);
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			//																					
			// Converting TO	pci8[8], pcu8[8], pci16[8], pcu16[8]			
			//						pci32[4], pcu32[4], pci64[2], pcu64[2]			
			//																					
			if constexpr (CT::SignedInteger8<TT> && S <= 8) {
				// double[8] -> pci8[8]													
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				auto
				i = simde_mm256_cvtps_epi32(f);
				i = simde_mm256_packs_epi32(i, i);
				i = simde_mm256_packs_epi16(i, i);
				return simde_mm256_castsi256_si128(i);
			}
			else if constexpr (CT::UnsignedInteger8<TT> && S <= 8) {
				// double[8] -> pcu8[8]													
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				auto
				i = simde_mm256_cvtps_epi32(f);
				i = simde_mm256_packus_epi32(i, i);
				i = simde_mm256_packus_epi16(i, i);
				return simde_mm256_castsi256_si128(i);
			}
			else if constexpr (CT::SignedInteger16<TT> && S <= 8) {
				// double[8] -> pci16[8]												
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				auto
				i = simde_mm256_cvtps_epi32(f);
				i = simde_mm256_packs_epi32(i, i);
				return simde_mm256_castsi256_si128(i);
			}
			else if constexpr (CT::UnsignedInteger16<TT> && S <= 8) {
				// double[8] -> pcu16[8]												
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				auto
				i = simde_mm256_cvtps_epi32(f);
				i = simde_mm256_packus_epi32(i, i);
				return simde_mm256_castsi256_si128(i);
			}
			else if constexpr (CT::Integer32<TT> && S <= 4) {
				// double[4] -> pci32[4]												
				// double[4] -> pcu32[4]												
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				return simde_mm256_castsi256_si128(simde_mm256_cvtps_epi32(f));
			}
			else if constexpr (CT::Integer64<TT> && S <= 2) {
				// double[2] -> pci64[2]												
				// double[2] -> pcu64[2]												
				const auto
				f = simde_mm512_castps512_ps256(simde_mm512_castpd_ps(v));
				auto
				i = simde_mm256_cvtps_epi32(f);
				i = simde_mm256_cvtepi32_epi64(simde_mm256_castsi256_si128(i));
				return simde_mm256_castsi256_si128(i);
			}
			else LANGULUS_ASSERT("Can't convert from __m512d to __m128i");
		}
		else LANGULUS_ASSERT("Can't convert from __m512d to unsupported");
	}

	/// Convert __m512i to any other register												
	///	@tparam TT - the true type contained in the result							
	///	@tparam S - size of the input array												
	///	@tparam FT - true type contained in the input								
	///	@tparam TO - type of register to convert to									
	///	@param v - the input data															
	///	@return the resulting register													
	template<class TT, Count S, class FT, class TO>
	auto ConvertFrom512i(const simde__m512i& v) noexcept {
		//																						
		// Converting FROM pci8[64], pcu8[64], pci16[32], pcu16[32]			
		//						 pci32[16], pcu32[16], pci64[8], pcu64[8]			
		//																						
		if constexpr (CT::Same<TO, simde__m128>) {
			//																					
			// Converting TO float[4]													
			//																					
			if constexpr (CT::Integer8<FT> && S <= 4) {
				// pci8[4] -> float[4]													
				// pcu8[4] -> float[4]													
				auto
				i16_32 = simde_mm512_castsi512_si128(v);
				i16_32 = simde_mm_unpacklo_epi8(i16_32, simde_mm_setzero_si128());
				i16_32 = simde_mm_unpacklo_epi16(i16_32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_ps(i16_32);
			}
			else if constexpr (CT::Integer16<FT> && S <= 4) {
				// pci16[4] -> float[4]													
				// pcu16[4] -> float[4]													
				auto
				i32 = simde_mm512_castsi512_si128(v);
				i32 = simde_mm_unpacklo_epi16(i32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_ps(i32);
			}
			else if constexpr (CT::Integer32<FT> && S <= 4) {
				// pci32[4] -> float[4]													
				// pcu32[4] -> float[4]													
				return simde_mm_cvtepi32_ps(simde_mm512_castsi512_si128(v));
			}
			else if constexpr (CT::Integer64<FT> && S <= 2) {
				// pci64[2] -> float[2]													
				// pcu64[2] -> float[2]													
				auto i32 = simde_mm512_cvtsepi64_epi32(v);
				return simde_mm_cvtepi32_ps(simde_mm256_castsi256_si128(i32));
			}
			else LANGULUS_ASSERT("Can't convert from __m512i to __m128");
		}
		else if constexpr (CT::Same<TO, simde__m128d>) {
			//																					
			// Converting TO double[2]													
			//																					
			if constexpr (CT::Integer8<FT> && S <= 4) {
				// pci8[4] -> float[4]													
				// pcu8[4] -> float[4]													
				auto
				i16_32 = simde_mm512_castsi512_si128(v);
				i16_32 = simde_mm_unpacklo_epi8(i16_32, simde_mm_setzero_si128());
				i16_32 = simde_mm_unpacklo_epi16(i16_32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_pd(i16_32);
			}
			else if constexpr (CT::Integer16<FT> && S <= 4) {
				// pci16[4] -> float[4]													
				// pcu16[4] -> float[4]													
				auto
				i32 = simde_mm512_castsi512_si128(v);
				i32 = simde_mm_unpacklo_epi16(i32, simde_mm_setzero_si128());
				return simde_mm_cvtepi32_pd(i32);
			}
			else if constexpr (CT::Integer32<FT> && S <= 4) {
				// pci32[4] -> float[4]													
				// pcu32[4] -> float[4]													
				return simde_mm_cvtepi32_pd(simde_mm512_castsi512_si128(v));
			}
			else if constexpr (CT::Integer64<FT> && S <= 2) {
				// pci64[2] -> float[2]													
				// pcu64[2] -> float[2]													
				auto i32 = simde_mm512_cvtsepi64_epi32(v);
				return simde_mm_cvtepi32_pd(simde_mm256_castsi256_si128(i32));
			}
			else LANGULUS_ASSERT("Can't convert from __m512i to __m128d");
		}
		else if constexpr (CT::Same<TO, simde__m128i>) {
			if constexpr (CT::Integer8<FT>) {
				//																				
				// Converting TO	pci8[16], pcu8[16], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 16)
					return simde_mm512_castsi512_si128(v);
				else if constexpr (CT::Integer16<TT> && S <= 8)
					return simde_mm_unpacklo_epi8(simde_mm512_castsi512_si128(v), simde_mm_setzero_si128());
				else if constexpr (CT::Integer32<TT> && S <= 4) {
					auto v16 = simde_mm_unpacklo_epi8(simde_mm512_castsi512_si128(v), simde_mm_setzero_si128());
					return simde_mm_unpacklo_epi16(v16, simde_mm_setzero_si128());
				}
				else if constexpr (CT::Integer64<TT> && S <= 2) {
					auto v16_32 = simde_mm_unpacklo_epi8(simde_mm512_castsi512_si128(v), simde_mm_setzero_si128());
					v16_32 = simde_mm_unpacklo_epi16(v16_32, simde_mm_setzero_si128());
					return simde_mm_unpacklo_epi32(v16_32, simde_mm_setzero_si128());
				}
				else LANGULUS_ASSERT("Can't convert from pci8/pcu8 to unsupported TT");
			}
			else if constexpr (CT::Integer16<FT>) {
				//																				
				// Converting TO	pci8[16], pcu8[16], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 16)
					return simde_mm256_castsi256_si128(simde_mm512_cvtsepi16_epi8(v));
				else if constexpr (CT::Integer16<TT> && S <= 8)
					return simde_mm512_castsi512_si128(v);
				else if constexpr (CT::Integer32<TT> && S <= 4)
					return simde_mm_unpacklo_epi16(simde_mm512_castsi512_si128(v), simde_mm_setzero_si128());
				else if constexpr (CT::Integer64<TT> && S <= 2) {
					auto v32 = simde_mm_unpacklo_epi16(simde_mm512_castsi512_si128(v), simde_mm_setzero_si128());
					return simde_mm_unpacklo_epi32(v32, simde_mm_setzero_si128());
				}
				else LANGULUS_ASSERT("Can't convert from pci16/pcu16 to unsupported TT");
			}
			else if constexpr (CT::Integer32<FT>) {
				//																				
				// Converting TO	pci8[16], pcu8[16], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 16)
					return simde_mm512_cvtsepi32_epi8(v);
				else if constexpr (CT::Integer16<TT> && S <= 8)
					return simde_mm256_castsi256_si128(simde_mm512_cvtsepi32_epi16(v));
				else if constexpr (CT::Integer32<TT> && S <= 4)
					return simde_mm512_castsi512_si128(v);
				else if constexpr (CT::Integer64<TT> && S <= 2)
					return simde_mm_cvtepi32_epi64(simde_mm512_castsi512_si128(v));
				else LANGULUS_ASSERT("Can't convert from pci32/pcu32 to unsupported TT");
			}
			else if constexpr (CT::Integer64<FT>) {
				//																				
				// Converting TO	pci8[8], pcu8[8], pci16[8], pcu16[8]		
				//						pci32[4], pcu32[4], pci64[2], pcu64[2]		
				//																				
				if constexpr (CT::Integer8<TT> && S <= 8)
					return simde_mm512_cvtsepi64_epi8(v);
				else if constexpr (CT::Integer16<TT> && S <= 8)
					return simde_mm512_cvtsepi64_epi16(v);
				else if constexpr (CT::Integer32<TT> && S <= 4)
					return simde_mm256_castsi256_si128(simde_mm512_cvtsepi64_epi32(v));
				else if constexpr (CT::Integer64<TT> && S <= 2)
					return simde_mm512_castsi512_si128(v);
				else LANGULUS_ASSERT("Can't convert from pci64/pcu64 to unsupported TT");
			}
			else LANGULUS_ASSERT("Can't convert from __m512i to __m128i");
		}
		else LANGULUS_ASSERT("Can't convert from __m512i to unsupported");
	}

} // namespace Langulus::SIMD