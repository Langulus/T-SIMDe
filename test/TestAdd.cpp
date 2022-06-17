///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#include "Main.hpp"
#include <catch2/catch.hpp>

template<class T1, class T2>
CT::Lossless<T1, T2> Control(const T1& lhs, const T2& rhs) noexcept {
	if constexpr (CT::Same<CT::Lossless<T1, T2>, ::std::byte>) {
		return static_cast<CT::Lossless<T1, T2>>(
			reinterpret_cast<const unsigned char&>(DenseCast(lhs)) +
			reinterpret_cast<const unsigned char&>(DenseCast(rhs))
		);
	}
	else return DenseCast(lhs) + DenseCast(rhs);
}

template<class T1, class T2, size_t C>
auto Control(const T1(&lhsArray)[C], const T2(&rhsArray)[C]) noexcept {
	using RT = CT::Lossless<T1, T2>;
	::std::array<RT, C> result;
	auto r = result.data();
	auto lhs = lhsArray;
	auto rhs = rhsArray;
	const auto lhsEnd = lhs + C;
	const auto rhsEnd = rhs + C;
	while (lhs != lhsEnd) {
		if constexpr (CT::Same<RT, ::std::byte>) {
			*r = static_cast<RT>(
				reinterpret_cast<const unsigned char&>(DenseCast(*lhs)) +
				reinterpret_cast<const unsigned char&>(DenseCast(*rhs))
			);
		}
		else *r = DenseCast(*lhs) + DenseCast(*rhs);
		++lhs;
		++rhs;
		++r;
	}

	return result;
}

TEMPLATE_TEST_CASE("Add", "[add]", SIGNED_TYPES(), UNSIGNED_TYPES(), SPARSE_SIGNED_TYPES(), SPARSE_UNSIGNED_TYPES()) {
	using T = TestType;

	GIVEN("scalar + scalar = scalar") {
		const T x = T{ 1 };
		const T y = Neg<T>(5);
		T r;
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[1] + vector[1] = vector[1]") {
		const T x[1] = { T{1} };
		const T y[1] = { Neg<T>(5) };
		T r[1];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 1; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 1; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[2] + vector[2] = vector[2]") {
		const T x[2] = { T{1}, T{2} };
		const T y[2] = { Neg<T>(5), T{6} };
		T r[2];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 2; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 2; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[3] + vector[3] = vector[3]") {
		const T x[3] = { T{1}, T{2}, T{0} };
		const T y[3] = { Neg<T>(5), T{6}, Neg<T>(22) };
		T r[3];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for(int i = 0; i < 3; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 3; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[4] + vector[4] = vector[4]") {
		const T x[4] = { T{1}, T{2}, T{0}, T{66} };
		const T y[4] = { Neg<T>(5), T{6}, Neg<T>(22), T{2} };
		T r[4];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 4; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 4; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[7] + vector[7] = vector[7]") {
		const T x[7] = { T{1}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0} };
		const T y[7] = { Neg<T>(5), T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22) };
		T r[7];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 7; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 7; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[8] + vector[8] = vector[8]") {
		const T x[8] = { T{1}, T{2}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0} };
		const T y[8] = { Neg<T>(5), T{6}, T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22) };
		T r[8];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 8; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 8; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[15] + vector[15] = vector[15]") {
		const T x[15] = { T{1}, T{2}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0}, T{2}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0} };
		const T y[15] = { Neg<T>(5), T{6}, T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22), T{6}, T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22) };
		T r[15];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 15; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 15; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}

	GIVEN("vector[16] + vector[16] = vector[16]") {
		const T x[16] = { T{1}, T{2}, T{2}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0}, T{2}, T{2}, T{0}, T{66}, T{1}, T{2}, T{0} };
		const T y[16] = { Neg<T>(5), Neg<T>(5), T{6}, T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22), T{6}, T{6}, Neg<T>(22), T{2}, Neg<T>(5), T{6}, Neg<T>(22) };
		T r[16];
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 16; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 16; ++i)
					REQUIRE(r[i] == rCheck[i]);
			}
		}
	}
}