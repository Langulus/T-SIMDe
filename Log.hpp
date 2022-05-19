///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#pragma once
#include "Fill.hpp"
#include "Convert.hpp"

namespace Langulus::SIMD
{

	enum class LogStyle {
		Natural,
		Base10,
		Base1P,
		Base2,
		FlooredBase2
	};

	/// Get natural/base-10/1p/base-2/floor(log2(x)) logarithm values via SIMD	
	///	@tparam STYLE - the type of the log function									
	///	@tparam T - the type of the array element										
	///	@tparam S - the size of the array												
	///	@tparam REGISTER - the register type (deducible)							
	///	@param value - the array 															
	///	@return the logarithm values														
	template<LogStyle STYLE = LogStyle::Base10, CT::Number T, Count S, TSIMD REGISTER>
	REGISTER InnerLog(const REGISTER& value) noexcept {
		static_assert(CT::Real<T>, 
			"SIMD::InnerLog doesn't work for whole numbers");

		if constexpr (SIMD128<REGISTER>) {
			if constexpr (CT::Same<T, float>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm_log_ps(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm_log10_ps(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm_log1p_ps(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm_log2_ps(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm_logb_ps(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of float[4] package");
			}
			else if constexpr (CT::Same<T, double>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm_log_pd(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm_log10_pd(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm_log1p_pd(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm_log2_pd(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm_logb_pd(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of double[2] package");
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLog of 16-byte package");
		}
		else if constexpr (SIMD256<REGISTER>) {
			if constexpr (CT::Same<T, float>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm256_log_ps(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm256_log10_ps(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm256_log1p_ps(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm256_log2_ps(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm256_logb_ps(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of float[8] package");
			}
			else if constexpr (CT::Same<T, double>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm256_log_pd(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm256_log10_pd(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm256_log1p_pd(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm256_log2_pd(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm256_logb_pd(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of double[4] package");
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLog of 32-byte package");
		}
		else if constexpr (SIMD512<REGISTER>) {
			if constexpr (CT::Same<T, float>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm512_log_ps(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm512_log10_ps(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm512_log1p_ps(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm512_log2_ps(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm512_logb_ps(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of float[16] package");
			}
			else if constexpr (CT::Same<T, double>) {
				if constexpr (STYLE == LogStyle::Natural)
					return simde_mm512_log_pd(value);
				else if constexpr (STYLE == LogStyle::Base10)
					return simde_mm512_log10_pd(value);
				else if constexpr (STYLE == LogStyle::Base1P)
					return simde_mm512_log1p_pd(value);
				else if constexpr (STYLE == LogStyle::Base2)
					return simde_mm512_log2_pd(value);
				else if constexpr (STYLE == LogStyle::FlooredBase2)
					return simde_mm512_logb_pd(value);
				else LANGULUS_ASSERT("Unsupported style for SIMD::InnerLog of double[8] package");
			}
			else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLog of 64-byte package");
		}
		else LANGULUS_ASSERT("Unsupported type for SIMD::InnerLog");
	}

	template<LogStyle STYLE, CT::Number T, Count S>
	auto Log(const T(&value)[S]) noexcept {
		return InnerLog<STYLE, T, S>(Load<0>(value));
	}

} // namespace Langulus::SIMD