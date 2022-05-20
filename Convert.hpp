///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Load.hpp"

#if LANGULUS_SIMD(128BIT)
	#include "ConvertFrom128.hpp"
#endif

#if LANGULUS_SIMD(256BIT)
	#include "ConvertFrom256.hpp"
#endif

#if LANGULUS_SIMD(512BIT)
	#include "ConvertFrom512.hpp"
#endif

#if LANGULUS_COMPILER(GCC)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wignored-attributes"
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
	auto Convert(const FT(&in)[S]) noexcept {
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
	/// fallback to opFALL and return conventionally									
	///	@tparam DEF - default value to fill empty register regions				
	///					  useful against division-by-zero cases						
	///	@tparam S1 - size of LHS array (deducible)									
	///	@tparam S2 - size of RHS array (deducible)									
	///	@tparam LHS - left number type (deducible)									
	///	@tparam RHS - right number type (deducible)									
	///	@tparam OPSIMD - the SIMD operation to invoke (deducible)				
	///	@tparam OPFALL - the fallback operation to invoke (deducible)			
	///	@param lhs - left argument															
	///	@param rhs - right argument														
	///	@param opSIMD - the function to invoke											
	///	@param opFALL - the function to invoke											
	///	@return the result (either std::array, number, or register)				
	template<int DEF, class REGISTER, class LOSSLESS, CT::Number LHS, CT::Number RHS, class FSIMD, class FFALL>
	NOD() auto AttemptSIMD(const LHS& lhs, const RHS& rhs, FSIMD&& opSIMD, FFALL&& opFALL)
	requires (Invocable<FSIMD, REGISTER> && Invocable<FFALL, LOSSLESS>) {
		using OUTSIMD = ::std::invoke_result_t<FSIMD, REGISTER, REGISTER>;
		constexpr auto S = ResultSize<LHS, RHS>();
		if constexpr (S < 2 || IsNotSupported<REGISTER> || IsNotSupported<OUTSIMD>) {
			// Call the fallback routine if unsupported or size 1				
			return Fallback<LOSSLESS>(lhs, rhs, Forward<decltype(opFALL)>(opFALL));
		}
		else if constexpr (CT::Array<LHS> && CT::Array<RHS>) {
			// Both LHS and RHS are arrays, so wrap in registers				
			return opSIMD(
				SIMD::Convert<DEF, LOSSLESS>(reinterpret_cast<const Decay<LHS>(&)[S]>(lhs)),
				SIMD::Convert<DEF, LOSSLESS>(reinterpret_cast<const Decay<RHS>(&)[S]>(rhs))
			);
		}
		else if constexpr (CT::Array<LHS>) {
			// LHS is array, RHS is scalar											
			return opSIMD(
				SIMD::Convert<DEF, LOSSLESS>(lhs),
				SIMD::Fill<REGISTER>(static_cast<LOSSLESS>(rhs))
			);
		}
		else if constexpr (CT::Array<RHS>) {
			// LHS is scalar, RHS is array											
			return opSIMD(
				SIMD::Fill<REGISTER>(static_cast<LOSSLESS>(lhs)),
				SIMD::Convert<DEF, LOSSLESS>(rhs)
			);
		}
		else {
			// Both LHS and RHS are scalars											
			return Fallback<LOSSLESS>(lhs, rhs, Forward<decltype(opFALL)>(opFALL));
		}
	}
	
} // namespace Langulus::SIMD

#if LANGULUS_COMPILER(GCC)
	#pragma GCC diagnostic pop
#endif