///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#include "Main.hpp"
#include <catch2/catch.hpp>
#include <random>

using timer = Catch::Benchmark::Chronometer;
template<class T>
using uninitialized = Catch::Benchmark::storage_for<T>;
std::random_device rd;
std::mt19937 gen(rd());

template<class T, Count C>
struct alignas(Langulus::Alignment) Vector {
	T mArray[C];

	Vector() {
		for (auto& i : mArray) {
			if constexpr (CT::Sparse<T>) {
				using TD = Decay<T>;
				i = new TD {static_cast<TD>(gen() % 66)};
			}
			else i = static_cast<T>(gen() % 66);
		}
	}

	~Vector() {
		for (auto& i : mArray) {
			if constexpr (CT::Sparse<T>)
				delete i;
		}
	}

	bool operator == (const Vector& e) const noexcept {
		for (int i = 0; i < C; ++i)
			if (DenseCast(mArray[i]) != DenseCast(e.mArray[i]))
				return false;
		return true;
	}
};

template<class LHS, class RHS, class OUT>
LANGULUS(ALWAYSINLINE) void Control(const LHS& lhs, const RHS& rhs, OUT& out) noexcept {
	if constexpr (CT::Same<OUT, ::std::byte>) {
		out = static_cast<OUT>(
			reinterpret_cast<const unsigned char&>(DenseCast(lhs)) +
			reinterpret_cast<const unsigned char&>(DenseCast(rhs))
		);
	}
	else out = DenseCast(lhs) + DenseCast(rhs);
}

template<class LHS, class RHS, size_t C, class OUT>
LANGULUS(ALWAYSINLINE) void Control(const Vector<LHS, C>& lhsArray, const Vector<RHS, C>& rhsArray, Vector<OUT, C>& out) noexcept {
	auto r = out.mArray;
	auto lhs = lhsArray.mArray;
	auto rhs = rhsArray.mArray;
	const auto lhsEnd = lhs + C;
	while (lhs != lhsEnd) {
		if constexpr (CT::Same<OUT, ::std::byte>) {
			*r = static_cast<OUT>(
				reinterpret_cast<const unsigned char&>(DenseCast(*lhs)) +
				reinterpret_cast<const unsigned char&>(DenseCast(*rhs))
			);
		}
		else *r = DenseCast(*lhs) + DenseCast(*rhs);

		++lhs; ++rhs; ++r;
	}
}

TEMPLATE_TEST_CASE("Add", "[add]", SPARSE_UNSIGNED_TYPES(), SPARSE_SIGNED_TYPES(), SIGNED_TYPES(), UNSIGNED_TYPES()) {
	using T = TestType;
	using DenseT = Decay<TestType>;

	GIVEN("scalar + scalar = scalar") {
		T x, y;
		DenseT r, rCheck;
		InitOne(x, 1);
		InitOne(y, -5);

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<T> nx(meter.runs());
					for (auto& i : nx)
						InitOne(i, 1);

					some<T> ny(meter.runs());
					for (auto& i : ny)
						InitOne(i, 1);

					some<DenseT> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<T> nx(meter.runs());
					for (auto& i : nx)
						InitOne(i, 1);

					some<T> ny(meter.runs());
					for (auto& i : ny)
						InitOne(i, 1);

					some<DenseT> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i], ny[i], nr[i]);
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}

		if constexpr (CT::Sparse<T>) {
			delete x;
			delete y;
		}
	}

	GIVEN("vector[1] + vector[1] = vector[1]") {
		Vector<T, 1> x, y;
		Vector<DenseT, 1> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 1>> nx(meter.runs());
					some<Vector<T, 1>> ny(meter.runs());
					some<Vector<DenseT, 1>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 1>> nx(meter.runs());
					some<Vector<T, 1>> ny(meter.runs());
					some<Vector<DenseT, 1>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[2] + vector[2] = vector[2]") {
		Vector<T, 2> x, y;
		Vector<DenseT, 2> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 2>> nx(meter.runs());
					some<Vector<T, 2>> ny(meter.runs());
					some<Vector<DenseT, 2>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 2>> nx(meter.runs());
					some<Vector<T, 2>> ny(meter.runs());
					some<Vector<DenseT, 2>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[3] + vector[3] = vector[3]") {
		Vector<T, 3> x, y;
		Vector<DenseT, 3> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 3>> nx(meter.runs());
					some<Vector<T, 3>> ny(meter.runs());
					some<Vector<DenseT, 3>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 3>> nx(meter.runs());
					some<Vector<T, 3>> ny(meter.runs());
					some<Vector<DenseT, 3>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[4] + vector[4] = vector[4]") {
		Vector<T, 4> x, y;
		Vector<DenseT, 4> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 4>> nx(meter.runs());
					some<Vector<T, 4>> ny(meter.runs());
					some<Vector<DenseT, 4>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 4>> nx(meter.runs());
					some<Vector<T, 4>> ny(meter.runs());
					some<Vector<DenseT, 4>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[7] + vector[7] = vector[7]") {
		Vector<T, 7> x, y;
		Vector<DenseT, 7> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 7>> nx(meter.runs());
					some<Vector<T, 7>> ny(meter.runs());
					some<Vector<DenseT, 7>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 7>> nx(meter.runs());
					some<Vector<T, 7>> ny(meter.runs());
					some<Vector<DenseT, 7>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") { 
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[8] + vector[8] = vector[8]") {
		Vector<T, 8> x, y;
		Vector<DenseT, 8> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 8>> nx(meter.runs());
					some<Vector<T, 8>> ny(meter.runs());
					some<Vector<DenseT, 8>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 8>> nx(meter.runs());
					some<Vector<T, 8>> ny(meter.runs());
					some<Vector<DenseT, 8>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[15] + vector[15] = vector[15]") {
		Vector<T, 15> x, y;
		Vector<DenseT, 15> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 15>> nx(meter.runs());
					some<Vector<T, 15>> ny(meter.runs());
					some<Vector<DenseT, 15>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 15>> nx(meter.runs());
					some<Vector<T, 15>> ny(meter.runs());
					some<Vector<DenseT, 15>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}

	GIVEN("vector[16] + vector[16] = vector[16]") {
		Vector<T, 16> x, y;
		Vector<DenseT, 16> r, rCheck;

		WHEN("Added") {
			Control(x, y, rCheck);
			SIMD::Add(x.mArray, y.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}

			#ifdef LANGULUS_STD_BENCHMARK
				BENCHMARK_ADVANCED("Add (control)") (timer meter) {
					some<Vector<T, 16>> nx(meter.runs());
					some<Vector<T, 16>> ny(meter.runs());
					some<Vector<DenseT, 16>> nr(meter.runs());
					meter.measure([&](int i) {
						Control(nx[i], ny[i], nr[i]);
						return nr[i];
					});
				};

				BENCHMARK_ADVANCED("Add (SIMD)") (timer meter) {
					some<Vector<T, 16>> nx(meter.runs());
					some<Vector<T, 16>> ny(meter.runs());
					some<Vector<DenseT, 16>> nr(meter.runs());
					meter.measure([&](int i) {
						SIMD::Add(nx[i].mArray, ny[i].mArray, nr[i].mArray);
						return nr[i];
					});
				};
			#endif
		}

		WHEN("Added in reverse") {
			Control(y, x, rCheck);
			SIMD::Add(y.mArray, x.mArray, r.mArray);

			THEN("The result should be correct") {
				REQUIRE(r == rCheck);
			}
		}
	}
}