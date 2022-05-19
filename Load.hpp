#pragma once
#include "SetGet.hpp"

namespace PCFW::Math::SIMD
{

	/// Wrap an array into a register														
	///	@tparam DEF - default number for setting elements outside S				
	///	@tparam T - the type of the array element (deducible)						
	///	@tparam S - the size of the array (deducible)								
	///	@param values - the array to load inside a register						
	///	@return the register																	
	template<int DEF, Number T, pcptr S>
	auto Load(const T(&v)[S]) noexcept {
		constexpr auto denseSize = sizeof(pcDecay<T>) * S;

		if constexpr (denseSize <= 16) {
			// Load as a single 128bit register										
			if constexpr (denseSize == 16 && Dense<T>) {
				if constexpr (IntegerNumber<T>)
					return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(v));
				else if constexpr (Same<T, pcr32>)
					return simde_mm_loadu_ps(v);
				else if constexpr (Same<T, pcr64>)
					return simde_mm_loadu_pd(v);
				else LANGULUS_ASSERT("Unsupported type for SIMD::Load 16-byte package");
			}
			else return Set<DEF, 16>(v);
		}
		else if constexpr (denseSize <= 32) {
			// Load as a single 256bit register										
			if constexpr (denseSize == 32 && Dense<T>) {
				if constexpr (IntegerNumber<T>)
					return simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(v));
				else if constexpr (Same<T, pcr32>)
					return simde_mm256_loadu_ps(v);
				else if constexpr (Same<T, pcr64>)
					return simde_mm256_loadu_pd(v);
				else LANGULUS_ASSERT("Unsupported type for SIMD::Load 32-byte package");
			}
			else return Set<DEF, 32>(v);
		}
		else if constexpr (denseSize <= 64) {
			// Load as a single 512bit register										
			if constexpr (denseSize == 64 && Dense<T>) {
				if constexpr (IntegerNumber<T>)
					return simde_mm512_loadu_si512(v);
				else if constexpr (Same<T, pcr32>)
					return simde_mm512_loadu_ps(v);
				else if constexpr (Same<T, pcr64>)
					return simde_mm512_loadu_pd(v);
				else LANGULUS_ASSERT("Unsupported type for SIMD::Load 64-byte package");
			}
			else return Set<DEF, 64>(v);
		}
		else LANGULUS_ASSERT("Unsupported array size for SIMD::Load");
	}

	
	/// Determine a SIMD register type that can wrap LHS and RHS					
	template<Number LHS, Number RHS>
	using TRegister = Conditional<
		((pcExtentOf<LHS>) > (pcExtentOf<RHS>)),
		decltype(SIMD::Load<0>(std::declval<TLossless<LHS, RHS>[pcExtentOf<LHS>]>())),
		decltype(SIMD::Load<0>(std::declval<TLossless<LHS, RHS>[pcExtentOf<RHS>]>()))
	>;

} // namespace PCFW::Math::SIMD