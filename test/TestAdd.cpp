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

		++lhs; ++rhs; ++r;
	}

	return result;
}

TEMPLATE_TEST_CASE("Add", "[add]", SIGNED_TYPES(), UNSIGNED_TYPES(), SPARSE_SIGNED_TYPES(), SPARSE_UNSIGNED_TYPES()) {
	using T = TestType;
	using DenseT = Decay<TestType>;

	GIVEN("scalar + scalar = scalar") {
		T x, y;
		DenseT r;
		InitOne(x, 1);
		InitOne(y, -5);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == rCheck);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(DenseCast(r) == rCheck);
			}
		}
	}

	GIVEN("vector[1] + vector[1] = vector[1]") {
		T x[1], y[1];
		DenseT r[1];
		Init(x, 1);
		Init(y, -5);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 1; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 1; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[2] + vector[2] = vector[2]") {
		T x[2], y[2];
		DenseT r[2];
		Init(x, 1, 2);
		Init(y, -5, 6);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 2; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 2; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[3] + vector[3] = vector[3]") {
		T x[3], y[3];
		DenseT r[3];
		Init(x, 1, 2, 0);
		Init(y, -5, 6, -22);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for(int i = 0; i < 3; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 3; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[4] + vector[4] = vector[4]") {
		T x[4], y[4];
		DenseT r[4];
		Init(x, 1, 2, 0, 66);
		Init(y, -5, 6, -22, 2);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 4; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 4; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[7] + vector[7] = vector[7]") {
		T x[7], y[7];
		DenseT r[7];
		Init(x, 1, 2, 0, 66, 1, 2, 0);
		Init(y, -5, 6, -22, 2, -5, 6, -22);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 7; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") { 
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 7; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[8] + vector[8] = vector[8]") {
		T x[8], y[8];
		DenseT r[8];
		Init(x, 1, 2, 2, 0, 66, 1, 2, 0);
		Init(y, -5, 6, 6, -22, 2, -5, 6, -22);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 8; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 8; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[15] + vector[15] = vector[15]") {
		T x[15], y[15];
		DenseT r[15];
		Init(x, 1, 2, 2, 0, 66, 1, 2, 0, 2, 2, 0, 66, 1, 2, 0);
		Init(y, -5, 6, 6, -22, 2, -5, 6, -22, 6, 6, -22, 2, -5, 6, -22);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 15; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 15; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}

	GIVEN("vector[16] + vector[16] = vector[16]") {
		T x[16], y[16];
		DenseT r[16];
		Init(x, 1, 2, 2, 2, 0, 66, 1, 2, 0, 2, 2, 0, 66, 1, 2, 0);
		Init(y, -5, 6, 6, -22, 2, -5, 6, -22, 6, 6, 6, -22, 2, -5, 6, -22);
		const auto rCheck = Control(x, y);

		WHEN("Added") {
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 16; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		WHEN("Added in reverse") {
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				for (int i = 0; i < 16; ++i)
					REQUIRE(DenseCast(r[i]) == rCheck[i]);
			}
		}

		Free(x);
		Free(y);
	}
}