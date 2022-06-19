///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Load.hpp"

#include "IgnoreWarningsPush.inl"

#if LANGULUS_SIMD(128BIT)
	#include "ConvertFrom128.hpp"
#endif

#if LANGULUS_SIMD(256BIT)
	#include "ConvertFrom256.hpp"
#endif

#if LANGULUS_SIMD(512BIT)
	#include "ConvertFrom512.hpp"
#endif

namespace Langulus::SIMD
{

	/// Convert from one array to another using SIMD									
	///	@tparam DEF - default values for elements that are not loaded			
	///	@tparam TT - type to convert to													
	///	@tparam S - size of the source array											
	///	@tparam FT - type to convert from												
	///	@param in - the input data															
	///	@return the resulting register													
	template<int DEF, class TT, Count S, class FT>
	LANGULUS(ALWAYSINLINE) auto Convert(const FT(&in)[S]) noexcept {
		using FROM = decltype(Load<DEF>(Uneval<Decay<FT>[S]>()));
		using TO = decltype(Load<DEF>(Uneval<Decay<TT>[S]>()));
		const FROM loaded = Load<DEF>(in);

		if constexpr (CT::NotSupported<FROM> || CT::NotSupported<TO>)
			return CT::Inner::NotSupported{};
		else if constexpr (CT::Same<TT, FT>)
			return loaded;

		#if LANGULUS_SIMD(128BIT)
			else if constexpr (CT::Same<FROM, simde__m128>)
				return ConvertFrom128<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m128d>)
				return ConvertFrom128d<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m128i>)
				return ConvertFrom128i<TT, S, FT, TO>(loaded);
		#endif

		#if LANGULUS_SIMD(256BIT)
			else if constexpr (CT::Same<FROM, simde__m256>)
				return ConvertFrom256<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m256d>)
				return ConvertFrom256d<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m256i>)
				return ConvertFrom256i<TT, S, FT, TO>(loaded);
		#endif

		#if LANGULUS_SIMD(512BIT)
			else if constexpr (CT::Same<FROM, simde__m512>)
				return ConvertFrom512<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m512d>)
				return ConvertFrom512d<TT, S, FT, TO>(loaded);
			else if constexpr (CT::Same<FROM, simde__m512i>)
				return ConvertFrom512i<TT, S, FT, TO>(loaded);
		#endif

		else LANGULUS_ASSERT("Can't convert from unsupported");
	}
	
	/// Attempt register encapsulation of LHS and RHS arrays							
	/// Check if result of opSIMD is supported and return it, otherwise			
	/// fallback to opFALL and calculate conventionally								
	///	@tparam DEF - default value to fill empty register regions				
	///					  useful against division-by-zero cases						
	///	@tparam REGISTER - the register to use for the SIMD operation			
	///	@tparam LOSSLESS - the type of data to use for the fallback				
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@tparam FSIMD - the SIMD operation to invoke (deducible)					
	///	@tparam FFALL - the fallback operation to invoke (deducible)			
	///	@param lhs - left argument															
	///	@param rhs - right argument														
	///	@param opSIMD - the function to invoke											
	///	@param opFALL - the function to invoke											
	///	@return the result (either std::array, number, or register)				
	template<int DEF, class REGISTER, class LOSSLESS, class LHS, class RHS, class FSIMD, class FFALL>
	LANGULUS(ALWAYSINLINE) NOD() auto AttemptSIMD(const LHS& lhs, const RHS& rhs, FSIMD&& opSIMD, FFALL&& opFALL) requires (Invocable<FSIMD, REGISTER> && Invocable<FFALL, LOSSLESS>) {
		using OUTSIMD = InvocableResult<FSIMD, REGISTER>;
		constexpr auto S = OverlapCount<LHS, RHS>();

		if constexpr (S < 2 || CT::NotSupported<REGISTER> || CT::NotSupported<OUTSIMD>) {
			// Call the fallback routine if unsupported or size 1				
			return Fallback<LOSSLESS>(lhs, rhs, Move(opFALL));
		}
		else if constexpr (CT::Array<LHS> && CT::Array<RHS>) {
			// Both LHS and RHS are arrays, so wrap in registers				
			return opSIMD(
				Convert<DEF, LOSSLESS, S>(lhs),
				Convert<DEF, LOSSLESS, S>(rhs)
			);
		}
		else if constexpr (CT::Array<LHS>) {
			// LHS is array, RHS is scalar											
			return opSIMD(
				Convert<DEF, LOSSLESS>(lhs),
				Fill<REGISTER>(static_cast<LOSSLESS>(rhs))
			);
		}
		else if constexpr (CT::Array<RHS>) {
			// LHS is scalar, RHS is array											
			return opSIMD(
				Fill<REGISTER>(static_cast<LOSSLESS>(lhs)),
				Convert<DEF, LOSSLESS>(rhs)
			);
		}
		else {
			// Both LHS and RHS are scalars											
			return Fallback<LOSSLESS>(lhs, rhs, Move(opFALL));
		}
	}
	
} // namespace Langulus::SIMD

#include "IgnoreWarningsPop.inl"
