///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "SetGet.hpp"

namespace Langulus::SIMD
{

	/// Wrap an array into a register														
	///	@tparam DEF - default number for setting elements outside S				
	///	@tparam T - the type of the array element (deducible)						
	///	@tparam S - the size of the array (deducible)								
	///	@param values - the array to load inside a register						
	///	@return the register																	
	template<int DEF, CT::Number T, Count S>
	auto Load(const T(&v)[S]) noexcept {
		constexpr auto denseSize = sizeof(pcDecay<T>) * S;

		if constexpr (denseSize <= 16) {
			// Load as a single 128bit register										
			if constexpr (denseSize == 16 && Dense<T>) {
				if constexpr (IntegerNumber<T>)
					return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(v));
				else if constexpr (CT::Same<T, float>)
					return simde_mm_loadu_ps(v);
				else if constexpr (CT::Same<T, double>)
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
				else if constexpr (CT::Same<T, float>)
					return simde_mm256_loadu_ps(v);
				else if constexpr (CT::Same<T, double>)
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
				else if constexpr (CT::Same<T, float>)
					return simde_mm512_loadu_ps(v);
				else if constexpr (CT::Same<T, double>)
					return simde_mm512_loadu_pd(v);
				else LANGULUS_ASSERT("Unsupported type for SIMD::Load 64-byte package");
			}
			else return Set<DEF, 64>(v);
		}
		else LANGULUS_ASSERT("Unsupported array size for SIMD::Load");
	}

} // namespace Langulus::SIMD

namespace Langulus::CT
{

	/// Determine a SIMD register type that can wrap LHS and RHS					
	template<CT::Number LHS, CT::Number RHS>
	using Register = Conditional<
		(ExtentOf<LHS> > ExtentOf<RHS>),
		decltype(SIMD::Load<0>(Uneval<Lossless<LHS, RHS>[ExtentOf<LHS>]>())),
		decltype(SIMD::Load<0>(Uneval<Lossless<LHS, RHS>[ExtentOf<RHS>]>()))
	>;

} // namespace Langulus::CT