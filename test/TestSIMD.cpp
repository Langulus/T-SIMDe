#include "Main.hpp"
#include <catch2/catch.hpp>


TEMPLATE_TEST_CASE("Symmetric signed SIMD operations", "[SIMD]", SIGNED_TYPES) {
	///																								
	GIVEN("Two numbers and a result") {
		using T = TestType;
		T x, y, r;

		WHEN("The numbers are compared for equality") {
			x = 1;
			y = 1;
			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			x = 1;
			y = -2;
			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are added") {
			x = 1;
			y = -5;
			SIMD::Add(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-4)));
			}
		}

		WHEN("The numbers are added in reverse") {
			x = 1;
			y = -5;
			SIMD::Add(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-4)));
			}
		}

		WHEN("The numbers are subtracted") {
			x = 90;
			y = -21;
			SIMD::Subtract(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(111)));
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			x = 90;
			y = -21;
			SIMD::Subtract(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-111)));
			}
		}

		WHEN("The numbers are multiplied") {
			x = 23;
			y = -3;
			SIMD::Multiply(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-69)));
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			x = 23;
			y = -3;
			SIMD::Multiply(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-69)));
			}
		}

		WHEN("The numbers are divided") {
			x = 33;
			y = -3;
			SIMD::Divide(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-11)));
			}
		}

		WHEN("The numbers are divided in reverse") {
			x = 33;
			y = -3;
			SIMD::Divide(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-3) / T(33)));
			}
		}

		/*WHEN("The numbers are divided by zero") {
			x = 33;
			y = 0;
			THEN("The operation should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}*/

		WHEN("The numbers are min'd") {
			x = 33;
			y = -3;
			SIMD::Min(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-3)));
			}
		}

		WHEN("The numbers are min'd in reverse") {
			x = 33;
			y = -3;
			SIMD::Min(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(-3)));
			}
		}

		WHEN("The numbers are max'd") {
			x = 33;
			y = -3;
			SIMD::Max(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(33)));
			}
		}

		WHEN("The numbers are max'd in reverse") {
			x = 33;
			y = -3;
			SIMD::Max(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r, T(33)));
			}
		}
	}

	///																								
	GIVEN("Two arrays[2] of numbers and a result[2]") {
		using T = TestType;
		T r[2];

		WHEN("The numbers are compared for equality") {
			T x[2] = { 1, 2 };
			T y[2] = { 1, 2 };
			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			T x[2] = { 1, 2 };
			T y[2] = { -5, 6 };
			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are added") {
			T x[2] = { 1, 2 };
			T y[2] = { -5, 6 };
			SIMD::Add(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r[0], T(-4)));
				REQUIRE(pcNear(r[1], T(8)));
			}
		}

		WHEN("The numbers are added in reverse") {
			T x[2] = { 1, 2 };
			T y[2] = { -5, 6 };
			SIMD::Add(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r[0], T(-4)));
				REQUIRE(pcNear(r[1], T(8)));
			}
		}

		WHEN("The numbers are subtracted") {
			T x[2] = { 120, 101 };
			T y[2] = { 21, -23 };
			SIMD::Subtract(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(120) - T(21)));
				REQUIRE(pcNear<T>(r[1], T(101) - T(-23)));
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			T x[2] = { 120, 101 };
			T y[2] = { 21, -23 };
			SIMD::Subtract(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(21) - T(120)));
				REQUIRE(pcNear<T>(r[1], T(-23) - T(101)));
			}
		}

		WHEN("The numbers are multiplied") {
			T x[2] = { 23, -24 };
			T y[2] = { 3, 4 };
			SIMD::Multiply(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(23) * T(3)));
				REQUIRE(pcNear<T>(r[1], T(-24) * T(4)));
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			T x[2] = { 23, -24 };
			T y[2] = { 3, 4 };
			SIMD::Multiply(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(23) * T(3)));
				REQUIRE(pcNear<T>(r[1], T(-24) * T(4)));
			}
		}

		WHEN("The numbers are divided") {
			T x[2] = { 33, 55 };
			T y[2] = { 3, -5 };
			SIMD::Divide(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33) / T(3)));
				REQUIRE(pcNear<T>(r[1], T(55) / T(-5)));
			}
		}

		WHEN("The numbers are divided in reverse") {
			T x[2] = { 33, 55 };
			T y[2] = { 3, -5 };
			SIMD::Divide(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3) / T(33)));
				REQUIRE(pcNear<T>(r[1], T(-5) / T(55)));
			}
		}

		/*WHEN("The numbers are divided by zero") {
			T x[2] = { 33, 55 };
			T y[2] = { 3, 0 };
			THEN("The division should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}*/

		WHEN("The numbers are min'd") {
			T x[2] = { 33, -5 };
			T y[2] = { 3, 55 };
			SIMD::Min(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3)));
				REQUIRE(pcNear<T>(r[1], T(-5)));
			}
		}

		WHEN("The numbers are min'd in reverse") {
			T x[2] = { 33, -5 };
			T y[2] = { 3, 55 };
			SIMD::Min(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3)));
				REQUIRE(pcNear<T>(r[1], T(-5)));
			}
		}

		WHEN("The numbers are max'd") {
			T x[2] = { 33, -5 };
			T y[2] = { 3, 55 };
			SIMD::Max(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33)));
				REQUIRE(pcNear<T>(r[1], T(55)));
			}
		}

		WHEN("The numbers are max'd in reverse") {
			T x[2] = { 33, -5 };
			T y[2] = { 3, 55 };
			SIMD::Max(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33)));
				REQUIRE(pcNear<T>(r[1], T(55)));
			}
		}
	}

	///																								
	GIVEN("Two arrays[3] of numbers and a result[3]") {
		using T = TestType;
		T r[3];

		WHEN("The numbers are compared for equality") {
			T x[3] = { -5, 6, 32 };
			T y[3] = { -5, 6, 32 };
			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			T x[3] = { 1, 2, -16 };
			T y[3] = { -5, 6, 32 };
			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are added") {
			T x[3] = { 1, 2, -16 };
			T y[3] = { -5, 6, 32 };
			SIMD::Add(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r[0], T(1) + T(-5)));
				REQUIRE(pcNear(r[1], T(2) + T(6)));
				REQUIRE(pcNear(r[2], T(-16) + T(32)));
			}
		}

		WHEN("The numbers are added in reverse") {
			T x[3] = { 1, 2, -16 };
			T y[3] = { -5, 6, 32 };
			SIMD::Add(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear(r[0], T(1) + T(-5)));
				REQUIRE(pcNear(r[1], T(2) + T(6)));
				REQUIRE(pcNear(r[2], T(-16) + T(32)));
			}
		}

		WHEN("The numbers are subtracted") {
			T x[3] = { 66, 101, -2 };
			T y[3] = { 21, -23, 0 };
			SIMD::Subtract(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(66) - T(21)));
				REQUIRE(pcNear<T>(r[1], T(101) - T(-23)));
				REQUIRE(pcNear<T>(r[2], T(-2) - T(0)));
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			T x[3] = { 66, 101, -2 };
			T y[3] = { 21, -23, 0 };
			SIMD::Subtract(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(21) - T(66)));
				REQUIRE(pcNear<T>(r[1], T(-23) - T(101)));
				REQUIRE(pcNear<T>(r[2], T(0) - T(-2)));
			}
		}

		WHEN("The numbers are multiplied") {
			T x[3] = { 23, -24, 0 };
			T y[3] = { 3, 4, 120 };
			SIMD::Multiply(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(23) * T(3)));
				REQUIRE(pcNear<T>(r[1], T(-24) * T(4)));
				REQUIRE(pcNear<T>(r[2], T(0) * T(120)));
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			T x[3] = { 23, -24, 0 };
			T y[3] = { 3, 4, 120 };
			SIMD::Multiply(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(23) * T(3)));
				REQUIRE(pcNear<T>(r[1], T(-24) * T(4)));
				REQUIRE(pcNear<T>(r[2], T(0) * T(120)));
			}
		}

		WHEN("The numbers are divided") {
			T x[3] = { 33, 55, 11 };
			T y[3] = { 3, -5, -1 };
			SIMD::Divide(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33) / T(3)));
				REQUIRE(pcNear<T>(r[1], T(55) / T(-5)));
				REQUIRE(pcNear<T>(r[2], T(11) / T(-1)));
			}
		}

		WHEN("The numbers are divided in reverse") {
			T x[3] = { 33, 55, 11 };
			T y[3] = { 3, -5, -1 };
			SIMD::Divide(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3) / T(33)));
				REQUIRE(pcNear<T>(r[1], T(-5) / T(55)));
				REQUIRE(pcNear<T>(r[2], T(-1) / T(11)));
			}
		}

		/*WHEN("The numbers are divided by zero") {
			T x[3] = { 33, 55, 11 };
			T y[3] = { 3, -5, 0 };
			THEN("The division should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}*/

		WHEN("The numbers are min'd") {
			T x[3] = { 33, -5, 0 };
			T y[3] = { 3, 55, 12 };
			SIMD::Min(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3)));
				REQUIRE(pcNear<T>(r[1], T(-5)));
				REQUIRE(pcNear<T>(r[2], T(0)));
			}
		}

		WHEN("The numbers are min'd in reverse") {
			T x[3] = { 33, -5, 0 };
			T y[3] = { 3, 55, 12 };
			SIMD::Min(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(3)));
				REQUIRE(pcNear<T>(r[1], T(-5)));
				REQUIRE(pcNear<T>(r[2], T(0)));
			}
		}

		WHEN("The numbers are max'd") {
			T x[3] = { 33, -5, 0 };
			T y[3] = { 3, 55, 12 };
			SIMD::Max(x, y, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33)));
				REQUIRE(pcNear<T>(r[1], T(55)));
				REQUIRE(pcNear<T>(r[2], T(12)));
			}
		}

		WHEN("The numbers are max'd in reverse") {
			T x[3] = { 33, -5, 0 };
			T y[3] = { 3, 55, 12 };
			SIMD::Max(y, x, r);
			THEN("The result should be correct") {
				REQUIRE(pcNear<T>(r[0], T(33)));
				REQUIRE(pcNear<T>(r[1], T(55)));
				REQUIRE(pcNear<T>(r[2], T(12)));
			}
		}
	}
}