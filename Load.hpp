///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "SetGet.hpp"
#include "IgnoreWarningsPush.inl"

namespace Langulus::SIMD
{

	/// Wrap an array into a register														
	///	@tparam DEF - default number for setting elements outside S				
	///	@tparam T - the type of the array element (deducible)						
	///	@tparam S - the size of the array (deducible)								
	///	@param v - the array to load inside a register								
	///	@return the register																	
	template<int DEF, class T, Count S>
	LANGULUS(ALWAYSINLINE) auto Load(const T(&v)[S]) noexcept {
		constexpr auto denseSize = sizeof(Decay<T>) * S;

		#if LANGULUS_SIMD(128BIT)
			if constexpr (denseSize <= 16) {
				// Load as a single 128bit register									
				if constexpr (denseSize == 16 && CT::Dense<T>) {
					if constexpr (CT::Integer<T> || CT::Byte<T> || CT::Character<T>)
						return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(v));
					else if constexpr (CT::RealSP<T>)
						return simde_mm_loadu_ps(v);
					else if constexpr (CT::RealDP<T>)
						return simde_mm_loadu_pd(v);
					else
						LANGULUS_ASSERT("Unsupported type for SIMD::Load 16-byte package");
				}
				else return Set<DEF, 16>(v);
			}
			else
		#endif

		#if LANGULUS_SIMD(256BIT)
			if constexpr (denseSize <= 32) {
				// Load as a single 256bit register									
				if constexpr (denseSize == 32 && CT::Dense<T>) {
					if constexpr (CT::Integer<T> || CT::Byte<T> || CT::Character<T>)
						return simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(v));
					else if constexpr (CT::RealSP<T>)
						return simde_mm256_loadu_ps(v);
					else if constexpr (CT::RealDP<T>)
						return simde_mm256_loadu_pd(v);
					else
						LANGULUS_ASSERT("Unsupported type for SIMD::Load 32-byte package");
				}
				else return Set<DEF, 32>(v);
			}
			else
		#endif

		#if LANGULUS_SIMD(512BIT)
			if constexpr (denseSize <= 64) {
				// Load as a single 512bit register									
				if constexpr (denseSize == 64 && CT::Dense<T>) {
					if constexpr (CT::Integer<T> || CT::Byte<T> || CT::Character<T>)
						return simde_mm512_loadu_si512(v);
					else if constexpr (CT::RealSP<T>)
						return simde_mm512_loadu_ps(v);
					else if constexpr (CT::RealDP<T>)
						return simde_mm512_loadu_pd(v);
					else
						LANGULUS_ASSERT("Unsupported type for SIMD::Load 64-byte package");
				}
				else return Set<DEF, 64>(v);
			}
			else
		#endif

		return CT::Inner::NotSupported {};
	}

} // namespace Langulus::SIMD

namespace Langulus::CT
{

	/// Determine a SIMD register type that can wrap LHS and RHS					
	template<class LHS, class RHS>
	using Register = Conditional<
		(ExtentOf<LHS> > ExtentOf<RHS>),
		decltype(SIMD::Load<0>(Uneval<Lossless<LHS, RHS>[ExtentOf<LHS>]>())),
		decltype(SIMD::Load<0>(Uneval<Lossless<LHS, RHS>[ExtentOf<RHS>]>()))
	>;

} // namespace Langulus::CT

#include "IgnoreWarningsPop.inl"
