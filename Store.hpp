#pragma once
#include "Intrinsics.hpp"

#if LANGULUS_COMPILER_IS(GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace PCFW::Math::SIMD
{

	/// Save a register to memory																
	///	@tparam FROM - the register to save												
	///	@tparam T - the type of data to write											
	///	@tparam S - the number of elements to write									
	///	@param from - the source register												
	///	@param to - the destination array												
	template<TSIMD FROM, Number T, pcptr S>
	void Store(const FROM& from, T(&to)[S]) noexcept {
		static_assert(S > 1, 
			"Storing less than two elements is suboptimal - avoid SIMD operations on such arrays as a whole");
		constexpr pcptr toSize = sizeof(pcDecay<T>) * S;

		//																						
		// __m128*																			
		//																						
		if constexpr (Same<FROM, simde__m128>) {
			if constexpr (Dense<T> && toSize == 16) {
				// Save to a dense array												
				simde_mm_storeu_ps(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(16) pcr32 temp[4];
				simde_mm_store_ps(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for(pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = temp[i];
			}
		}
		else if constexpr (Same<FROM, simde__m128d>) {
			if constexpr (Dense<T> && toSize == 16) {
				// Save to a dense array												
				simde_mm_storeu_pd(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				simde_mm_storel_pd(pcPtr(to[0]), from);
				if constexpr (S > 1)
					simde_mm_storeh_pd(pcPtr(to[1]), from);
			}
		}
		else if constexpr (Same<FROM, simde__m128i>) {
			if constexpr (Dense<T> && toSize == 16) {
				// Save to a dense array												
				simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(to), from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(16) pcu8 temp[16];
				simde_mm_store_si128(reinterpret_cast<simde__m128i*>(temp), from);
				for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = reinterpret_cast<const pcDecay<T>*>(temp)[i];
			}
		}

		//																						
		// __m256*																			
		//																						
		else if constexpr (Same<FROM, simde__m256>) {
			if constexpr (Dense<T> && toSize == 32) {
				// Save to a dense array												
				simde_mm256_storeu_ps(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(32) pcr32 temp[8];
				simde_mm256_store_ps(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = temp[i];
			}
		}
		else if constexpr (Same<FROM, simde__m256d>) {
			if constexpr (Dense<T> && toSize == 32) {
				// Save to a dense array												
				simde_mm256_storeu_pd(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(32) pcr64 temp[4];
				simde_mm256_store_pd(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = temp[i];
			}
		}
		else if constexpr (Same<FROM, simde__m256i>) {
			if constexpr (Dense<T> && toSize == 32) {
				// Save to a dense array												
				simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(to), from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(32) pcu8 temp[32];
				simde_mm256_store_si256(reinterpret_cast<simde__m256i*>(temp), from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = reinterpret_cast<const pcDecay<T>*>(temp)[i];
			}
		}

		//																						
		// __m512*																			
		//																						
		else if constexpr (Same<FROM, simde__m512>) {
			if constexpr (Dense<T> && toSize == 64) {
				// Save to a dense array												
				simde_mm512_storeu_ps(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(64) pcr32 temp[16];
				simde_mm512_store_ps(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = temp[i];
			}
		}
		else if constexpr (Same<FROM, simde__m512d>) {
			if constexpr (Dense<T> && toSize == 64) {
				// Save to a dense array												
				simde_mm512_storeu_pd(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(64) pcr64 temp[8];
				simde_mm512_store_pd(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = temp[i];
			}
		}
		else if constexpr (Same<FROM, simde__m512i>) {
			if constexpr (Dense<T> && toSize == 64) {
				// Save to a dense array												
				simde_mm512_storeu_si512(to, from);
			}
			else {
				// Save to a sparse array, or a differently sized array		
				alignas(64) pcu8 temp[64];
				simde_mm512_store_si512(temp, from);
				if constexpr (Dense<T>)
					pcCopyMemory(temp, to, toSize);
				else for (pcptr i = 0; i < S; ++i)
					pcVal(to[i]) = reinterpret_cast<const pcDecay<T>*>(temp)[i];
			}
		}
		else LANGULUS_ASSERT("Unsupported FROM register for SIMD::Store");
	}

} // namespace PCFW::Math::SIMD

#if LANGULUS_COMPILER_IS(GCC)
#pragma GCC diagnostic pop
#endif