///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Intrinsics.hpp"
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

	/// Get an element of an array, or zero if out of range							
	///	@tparam DEF - the default value for the element, if out of S			
	///	@tparam IDX - the index to get													
	///	@tparam MAXS - the maximum number of elements T inside register		
	///	@tparam REVERSE - whether or not to count in inverse						
	///	@tparam T - the type of an array element										
	///	@tparam S - the size of the array 												
	///	@param values - the array to access												
	///	@return a reference to the element, or DEF if out of range				
	template<class R, int DEF, Offset IDX, Count MAXS, bool REVERSE = false, class T, Count S>
	LANGULUS(ALWAYSINLINE) const R& InnerGet(const T(&values)[S]) {
		static_assert(S <= MAXS, "S must be in MAXS limit");
		static_assert(IDX < MAXS, "IDX must be in MAXS limit");
		static constinit auto fallback = static_cast<R>(DEF);
		if constexpr (REVERSE) {
			if constexpr (MAXS - IDX - 1 < S)
				return reinterpret_cast<const R&>(DenseCast(values[MAXS - IDX - 1]));
			else
				return fallback;
		}
		else {
			if constexpr (IDX < S)
				return reinterpret_cast<const R&>(DenseCast(values[IDX]));
			else
				return fallback;
		}
	}

	/// Inner array iteration and register setting										
	///	@tparam DEF - the default value for the element, if out of S			
	///	@tparam CHUNK - the size of register to fill (in bytes)					
	///	@tparam T - the type of an array element										
	///	@tparam S - the size of the array 												
	///	@tparam INDICES - the indices to use											
	///	@param values - the array to access												
	///	@return the register																	
	template<int DEF, Size CHUNK, class T, Count S, Offset... INDICES>
	LANGULUS(ALWAYSINLINE) auto InnerSet(std::integer_sequence<Offset, INDICES...>, const T(&values)[S]) {
		#if LANGULUS_SIMD(128BIT)
			if constexpr (CHUNK == 16) {
				if constexpr (CT::SignedInteger8<T>)
					return simde_mm_setr_epi8(InnerGet<int8_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::UnsignedInteger8<T>)
					return simde_mm_setr_epi8(InnerGet<int8_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::SignedInteger16<T>)
					return simde_mm_setr_epi16(InnerGet<int16_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::UnsignedInteger16<T>)
					return simde_mm_setr_epi16(InnerGet<int16_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::SignedInteger32<T>)
					return simde_mm_setr_epi32(InnerGet<int32_t, DEF, INDICES, 4>(values)...);
				else if constexpr (CT::UnsignedInteger32<T>)
					return simde_mm_setr_epi32(InnerGet<int32_t, DEF, INDICES, 4>(values)...);
				else if constexpr (CT::SignedInteger64<T>)
					return simde_mm_set_epi64x(InnerGet<int64_t, DEF, INDICES, 2, true>(values)...);
				else if constexpr (CT::UnsignedInteger64<T>)
					return simde_mm_set_epi64x(InnerGet<int64_t, DEF, INDICES, 2, true>(values)...);
				else if constexpr (CT::RealSP<T>)
					return simde_mm_setr_ps(InnerGet<simde_float32, DEF, INDICES, 4>(values)...);
				else if constexpr (CT::RealDP<T>)
					return simde_mm_setr_pd(InnerGet<simde_float64, DEF, INDICES, 2>(values)...);
				else
					LANGULUS_ASSERT("Can't SIMD::InnerSet 16-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(256BIT)
			if constexpr (CHUNK == 32) {
				if constexpr (CT::SignedInteger8<T>)
					return simde_mm256_setr_epi8(InnerGet<int8_t, DEF, INDICES, 32>(values)...);
				else if constexpr (CT::UnsignedInteger8<T>)
					return simde_mm256_setr_epi8(InnerGet<int8_t, DEF, INDICES, 32>(values)...);
				else if constexpr (CT::SignedInteger16<T>)
					return simde_mm256_setr_epi16(InnerGet<int16_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::UnsignedInteger16<T>)
					return simde_mm256_setr_epi16(InnerGet<int16_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::SignedInteger32<T>)
					return simde_mm256_setr_epi32(InnerGet<int32_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::UnsignedInteger32<T>)
					return simde_mm256_setr_epi32(InnerGet<int32_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::SignedInteger64<T>)
					return simde_mm256_setr_epi64x(InnerGet<int64_t, DEF, INDICES, 4>(values)...);
				else if constexpr (CT::UnsignedInteger64<T>)
					return simde_mm256_setr_epi64x(InnerGet<int64_t, DEF, INDICES, 4>(values)...);
				else if constexpr (CT::RealSP<T>)
					return simde_mm256_setr_ps(InnerGet<simde_float32, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::RealDP<T>)
					return simde_mm256_setr_pd(InnerGet<simde_float64, DEF, INDICES, 4>(values)...);
				else
					LANGULUS_ASSERT("Can't SIMD::InnerSet 32-byte package");
			}
			else
		#endif

		#if LANGULUS_SIMD(512BIT)
			if constexpr (CHUNK == 64) {
				if constexpr (CT::SignedInteger8<T>)
					return simde_mm512_setr_epi8(InnerGet<int8_t, DEF, INDICES, 64>(values)...);
				else if constexpr (CT::UnsignedInteger8<T>)
					return simde_mm512_setr_epi8(InnerGet<int8_t, DEF, INDICES, 64>(values)...);
				else if constexpr (CT::SignedInteger16<T>)
					return simde_mm512_setr_epi16(InnerGet<int16_t, DEF, INDICES, 32>(values)...);
				else if constexpr (CT::UnsignedInteger16<T>)
					return simde_mm512_setr_epi16(InnerGet<int16_t, DEF, INDICES, 32>(values)...);
				else if constexpr (CT::SignedInteger32<T>)
					return simde_mm512_setr_epi32(InnerGet<int32_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::UnsignedInteger32<T>)
					return simde_mm512_setr_epi32(InnerGet<int32_t, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::SignedInteger64<T>)
					return simde_mm512_setr_epi64(InnerGet<int64_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::UnsignedInteger64<T>)
					return simde_mm512_setr_epi64(InnerGet<int64_t, DEF, INDICES, 8>(values)...);
				else if constexpr (CT::RealSP<T>)
					return simde_mm512_setr_ps(InnerGet<simde_float32, DEF, INDICES, 16>(values)...);
				else if constexpr (CT::RealDP<T>)
					return simde_mm512_setr_pd(InnerGet<simde_float64, DEF, INDICES, 8>(values)...);
				else
					LANGULUS_ASSERT("Can't SIMD::InnerSet 64-byte package");
			}
			else
		#endif

		LANGULUS_ASSERT("Unsupported package size for SIMD::InnerSet");
	}

	/// Construct a register manually														
	///	@tparam CHUNK - the size of the chunk to set									
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@param values - the array to wrap												
	///	@return the register																	
	template<int DEF, Size CHUNK, class T, Count S>
	LANGULUS(ALWAYSINLINE) auto Set(const T(&values)[S]) noexcept {
		if constexpr (S < 2) {
			// No point in storing a single value in a large register		
			// If this is reached, then the library did not choose the		
			// optimal route for your operation at compile time				
			return CT::Inner::NotSupported {};
		}
		else {
			constexpr auto MaxS = CHUNK / sizeof(Decay<T>);
			static_assert((CT::Dense<T> && MaxS > S) || (CT::Sparse<T> && MaxS >= S),
				"S should be smaller (or equal if sparse) than MaxS - use load otherwise");
			return InnerSet<DEF, CHUNK>(::std::make_integer_sequence<Count, MaxS>(), values);
		}
	}

} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
