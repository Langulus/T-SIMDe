#pragma once
#include "Intrinsics.hpp"

namespace PCFW::Math::SIMD
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
	template<int DEF, pcptr IDX, pcptr MAXS, bool REVERSE = false, Number T, pcptr S>
	const pcDecay<T>& InnerGet(const T(&values)[S]) {
		static_assert(S <= MAXS, "S must be in MAXS limit");
		static_assert(IDX < MAXS, "IDX must be in MAXS limit");
		static constinit auto fallback = static_cast<pcDecay<T>>(DEF);
		if constexpr (REVERSE) {
			if constexpr (MAXS - IDX - 1 < S)
				return pcVal(values[MAXS - IDX - 1]);
			else
				return fallback;
		}
		else {
			if constexpr (IDX < S)
				return pcVal(values[IDX]);
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
	template<int DEF, pcptr CHUNK, Number T, pcptr S, pcptr... INDICES>
	auto InnerSet(std::integer_sequence<pcptr, INDICES...>, const T(&values)[S]) {
		if constexpr (CHUNK == 16) {
			if constexpr (SignedInteger8<T>)
				return simde_mm_setr_epi8(InnerGet<DEF, INDICES, 16>(values)...);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm_setr_epi8(InnerGet<DEF, INDICES, 16>(reinterpret_cast<const pci8(&)[S]>(values))...);
			else if constexpr (SignedInteger16<T>)
				return simde_mm_setr_epi16(InnerGet<DEF, INDICES, 8>(values)...);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm_setr_epi16(InnerGet<DEF, INDICES, 8>(reinterpret_cast<const pci16(&)[S]>(values))...);
			else if constexpr (SignedInteger32<T>)
				return simde_mm_setr_epi32(InnerGet<DEF, INDICES, 4>(values)...);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm_setr_epi32(InnerGet<DEF, INDICES, 4>(reinterpret_cast<const pci32(&)[S]>(values))...);
			else if constexpr (SignedInteger64<T>)
				return simde_mm_set_epi64x(InnerGet<DEF, INDICES, 2, true>(values)...);
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm_set_epi64x(InnerGet<DEF, INDICES, 2, true>(reinterpret_cast<const pci64(&)[S]>(values))...);
			else if constexpr (Same<T, pcr32>)
				return simde_mm_setr_ps(InnerGet<DEF, INDICES, 4>(values)...);
			else if constexpr (Same<T, pcr64>)
				return simde_mm_setr_pd(InnerGet<DEF, INDICES, 2>(values)...);
			else LANGULUS_ASSERT("Can't SIMD::InnerSet 16-byte package");
		}
		else if constexpr (CHUNK == 32) {
			if constexpr (SignedInteger8<T>)
				return simde_mm256_setr_epi8(InnerGet<DEF, INDICES, 32>(values)...);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm256_setr_epi8(InnerGet<DEF, INDICES, 32>(reinterpret_cast<const pci8(&)[S]>(values))...);
			else if constexpr (SignedInteger16<T>)
				return simde_mm256_setr_epi16(InnerGet<DEF, INDICES, 16>(values)...);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm256_setr_epi16(InnerGet<DEF, INDICES, 16>(reinterpret_cast<const pci16(&)[S]>(values))...);
			else if constexpr (SignedInteger32<T>)
				return simde_mm256_setr_epi32(InnerGet<DEF, INDICES, 8>(values)...);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm256_setr_epi32(InnerGet<DEF, INDICES, 8>(reinterpret_cast<const pci32(&)[S]>(values))...);
			else if constexpr (SignedInteger64<T>)
				return simde_mm256_setr_epi64x(InnerGet<DEF, INDICES, 4>(values)...);
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm256_setr_epi64x(InnerGet<DEF, INDICES, 4>(reinterpret_cast<const pci64(&)[S]>(values))...);
			else if constexpr (Same<T, pcr32>)
				return simde_mm256_setr_ps(InnerGet<DEF, INDICES, 8>(values)...);
			else if constexpr (Same<T, pcr64>)
				return simde_mm256_setr_pd(InnerGet<DEF, INDICES, 4>(values)...);
			else LANGULUS_ASSERT("Can't SIMD::InnerSet 32-byte package");
		}
		else if constexpr (CHUNK == 64) {
			if constexpr (SignedInteger8<T>)
				return simde_mm512_setr_epi8(InnerGet<DEF, INDICES, 64>(values)...);
			else if constexpr (UnsignedInteger8<T>)
				return simde_mm512_setr_epi8(InnerGet<DEF, INDICES, 64>(reinterpret_cast<const pci8(&)[S]>(values))...);
			else if constexpr (SignedInteger16<T>)
				return simde_mm512_setr_epi16(InnerGet<DEF, INDICES, 32>(values)...);
			else if constexpr (UnsignedInteger16<T>)
				return simde_mm512_setr_epi16(InnerGet<DEF, INDICES, 32>(reinterpret_cast<const pci16(&)[S]>(values))...);
			else if constexpr (SignedInteger32<T>)
				return simde_mm512_setr_epi32(InnerGet<DEF, INDICES, 16>(values)...);
			else if constexpr (UnsignedInteger32<T>)
				return simde_mm512_setr_epi32(InnerGet<DEF, INDICES, 16>(reinterpret_cast<const pci32(&)[S]>(values))...);
			else if constexpr (SignedInteger64<T>)
				return simde_mm512_setr_epi64(InnerGet<DEF, INDICES, 8>(values)...);
			else if constexpr (UnsignedInteger64<T>)
				return simde_mm512_setr_epi64(InnerGet<DEF, INDICES, 8>(reinterpret_cast<const pci64(&)[S]>(values))...);
			else if constexpr (Same<T, pcr32>)
				return simde_mm512_setr_ps(InnerGet<DEF, INDICES, 16>(values)...);
			else if constexpr (Same<T, pcr64>)
				return simde_mm512_setr_pd(InnerGet<DEF, INDICES, 8>(values)...);
			else LANGULUS_ASSERT("Can't SIMD::InnerSet 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported package size for SIMD::InnerSet");
	}

	/// Construct a register manually														
	///	@tparam CHUNK - the size of the chunk to set									
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@param values - the array to wrap												
	///	@return the register																	
	template<int DEF, pcptr CHUNK, class T, pcptr S>
	auto Set(const T(&values)[S]) noexcept {
		if constexpr (S < 2)
			return SIMD::NotSupported{};
		else {
			constexpr auto MaxS = CHUNK / sizeof(pcDecay<T>);
			static_assert((Dense<T> && MaxS > S) || (Sparse<T> && MaxS >= S),
				"S should be smaller (or equal if sparse) than MaxS - use load otherwise");
			return InnerSet<DEF, CHUNK>(std::make_integer_sequence<pcptr, MaxS>(), values);
		}
	}

} // namespace PCFW::Math::SIMD