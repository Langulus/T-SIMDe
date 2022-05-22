///																									
/// Langulus::TSIMDe																				
/// Copyright(C) 2019 Dimo Markov <langulusteam@gmail.com>							
///																									
/// Distributed under GNU General Public License v3+									
/// See LICENSE file, or https://www.gnu.org/licenses									
///																									
#include "Main.hpp"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Symmetric signed SIMD operations", "[SIMD]", SIGNED_TYPES()) {
	GIVEN("Two numbers and a result") {
		using T = TestType;
		static_assert(CT::Signed<T>, "Test is incorrect, type must be signed");

		T x, y, r;

		/*WHEN("The numbers are compared for equality") {
			x = T {1};
			y = T {1};

			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			x = T {1};
			y = T {-2};

			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}*/

		WHEN("The numbers are added") {
			x = T {1};
			y = T {-5};
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-4});
			}
		}

		WHEN("The numbers are added in reverse") {
			x = T {1};
			y = T {-5};
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-4});
			}
		}

		WHEN("The numbers are subtracted") {
			x = T {90};
			y = T {-21};
			SIMD::Subtract(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {111});
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			x = T {90};
			y = T {-21};
			SIMD::Subtract(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-111});
			}
		}

		WHEN("The numbers are multiplied") {
			x = T {23};
			y = T {-3};
			SIMD::Multiply(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-69});
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			x = T {23};
			y = T {-3};
			SIMD::Multiply(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-69});
			}
		}

		WHEN("The numbers are divided") {
			x = T {33};
			y = T {-3};
			SIMD::Divide(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-11});
			}
		}

		WHEN("The numbers are divided in reverse") {
			x = T {33};
			y = T {-3};
			SIMD::Divide(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-3} / T {33});
			}
		}

		WHEN("The numbers are divided by zero") {
			x = T {33};
			y = T {0};

			THEN("The operation should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}

		WHEN("The numbers are min'd") {
			x = T {33};
			y = T {-3};
			SIMD::Min(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-3});
			}
		}

		WHEN("The numbers are min'd in reverse") {
			x = T {33};
			y = T {-3};
			SIMD::Min(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {-3});
			}
		}

		WHEN("The numbers are max'd") {
			x = T {33};
			y = T {-3};
			SIMD::Max(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {33});
			}
		}

		WHEN("The numbers are max'd in reverse") {
			x = T {33};
			y = T {-3};
			SIMD::Max(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r == T {33});
			}
		}
	}

	///																								
	GIVEN("Two arrays[2] of numbers and a result[2]") {
		using T = TestType;
		T r[2];

		/*WHEN("The numbers are compared for equality") {
			T x[2] = {T{1}, T{2}};
			T y[2] = {T{1}, T{2}};

			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			T x[2] = {T{1}, T{2}};
			T y[2] = {T{-5}, T{6}};

			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}*/

		WHEN("The numbers are added") {
			T x[2] = {T{1}, T{2}};
			T y[2] = {T{-5}, T{6}};
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {-4});
				REQUIRE(r[1] == T {8});
			}
		}

		WHEN("The numbers are added in reverse") {
			T x[2] = {T{1}, T{2}};
			T y[2] = {T{-5}, T{6}};
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {-4});
				REQUIRE(r[1] == T {8});
			}
		}

		WHEN("The numbers are subtracted") {
			T x[2] = {T{120}, T{101}};
			T y[2] = {T{21}, T{-23}};
			SIMD::Subtract(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {120} - T {21});
				REQUIRE(r[1] == T {101} - T {-23});
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			T x[2] = {T{120}, T{101}};
			T y[2] = {T{21}, T{-23}};
			SIMD::Subtract(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {21} - T {120});
				REQUIRE(r[1] == T {-23} - T {101});
			}
		}

		WHEN("The numbers are multiplied") {
			T x[2] = {T{23}, T{-24}};
			T y[2] = {T{3}, T{4}};
			SIMD::Multiply(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {23} * T {3});
				REQUIRE(r[1] == T {-24} * T {4});
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			T x[2] = {T{23}, T{-24}};
			T y[2] = {T{3}, T{4}};
			SIMD::Multiply(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {23} * T {3});
				REQUIRE(r[1] == T {-24} * T {4});
			}
		}

		WHEN("The numbers are divided") {
			T x[2] = {T{33}, T{55}};
			T y[2] = {T{3}, T{-5}};
			SIMD::Divide(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33} / T {3});
				REQUIRE(r[1] == T {55} / T {-5});
			}
		}

		WHEN("The numbers are divided in reverse") {
			T x[2] = {T{33}, T{55}};
			T y[2] = {T{3}, T{-5}};
			SIMD::Divide(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3} / T {33});
				REQUIRE(r[1] == T {-5} / T {55});
			}
		}

		WHEN("The numbers are divided by zero") {
			T x[2] = {T{33}, T{55}};
			T y[2] = {T{3}, T{0}};

			THEN("The division should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}

		WHEN("The numbers are min'd") {
			T x[2] = {T{33}, T{-5}};
			T y[2] = {T{3}, T{55}};
			SIMD::Min(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3});
				REQUIRE(r[1] == T {-5});
			}
		}

		WHEN("The numbers are min'd in reverse") {
			T x[2] = {T{33}, T{-5}};
			T y[2] = {T{3}, T{55}};
			SIMD::Min(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3});
				REQUIRE(r[1] == T {-5});
			}
		}

		WHEN("The numbers are max'd") {
			T x[2] = {T{33}, T{-5}};
			T y[2] = {T{3}, T{55}};
			SIMD::Max(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33});
				REQUIRE(r[1] == T {55});
			}
		}

		WHEN("The numbers are max'd in reverse") {
			T x[2] = {T{33}, T{-5}};
			T y[2] = {T{3}, T{55}};
			SIMD::Max(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33});
				REQUIRE(r[1] == T {55});
			}
		}
	}

	///																								
	GIVEN("Two arrays[3] of numbers and a result[3]") {
		using T = TestType;
		T r[3];

		/*WHEN("The numbers are compared for equality") {
			T x[3] = {T{-5}, T{6}, T{32}};
			T y[3] = {T{-5}, T{6}, T{32}};

			THEN("The result should be correct") {
				REQUIRE(SIMD::Equals(x, y));
			}
		}

		WHEN("The numbers are compared for inequality") {
			T x[3] = {T{1}, T{2}, T{-16}};
			T y[3] = {T{-5}, T{6}, T{32}};

			THEN("The result should be correct") {
				REQUIRE(!SIMD::Equals(x, y));
			}
		}*/

		WHEN("The numbers are added") {
			T x[3] = {T{1}, T{2}, T{-16}};
			T y[3] = {T{-5}, T{6}, T{32}};
			SIMD::Add(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {1} + T {-5});
				REQUIRE(r[1] == T {2} + T {6});
				REQUIRE(r[2] == T {-16} + T {32});
			}
		}

		WHEN("The numbers are added in reverse") {
			T x[3] = {T{1}, T{2}, T{-16}};
			T y[3] = {T{-5}, T{6}, T{32}};
			SIMD::Add(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {1} + T {-5});
				REQUIRE(r[1] == T {2} + T {6});
				REQUIRE(r[2] == T {-16} + T {32});
			}
		}

		WHEN("The numbers are subtracted") {
			T x[3] = {T{66}, T{101}, T{-2}};
			T y[3] = {T{21}, T{-23}, 0};
			SIMD::Subtract(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {66} - T {21});
				REQUIRE(r[1] == T {101} - T {-23});
				REQUIRE(r[2] == T {-2} - T {0});
			}
		}

		WHEN("The numbers are subtracted in reverse") {
			T x[3] = {T{66}, T{101}, T{-2}};
			T y[3] = {T{21}, T{-23}, 0};
			SIMD::Subtract(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {21} - T {66});
				REQUIRE(r[1] == T {-23} - T {101});
				REQUIRE(r[2] == T {0} - T {-2});
			}
		}

		WHEN("The numbers are multiplied") {
			T x[3] = {T{23}, T{-24}, T{0}};
			T y[3] = {T{3}, T{4}, T{120}};
			SIMD::Multiply(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {23} * T {3});
				REQUIRE(r[1] == T {-24} * T {4});
				REQUIRE(r[2] == T {0} * T {120});
			}
		}

		WHEN("The numbers are multiplied in reverse") {
			T x[3] = {T{23}, T{-24}, T{0}};
			T y[3] = {T{3}, T{4}, T{120}};
			SIMD::Multiply(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {23} * T {3});
				REQUIRE(r[1] == T {-24} * T {4});
				REQUIRE(r[2] == T {0} * T {120});
			}
		}

		WHEN("The numbers are divided") {
			T x[3] = {T{33}, T{55}, T{11}};
			T y[3] = {T{3}, T{-5}, T{-1}};
			SIMD::Divide(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33} / T {3});
				REQUIRE(r[1] == T {55} / T {-5});
				REQUIRE(r[2] == T {11} / T {-1});
			}
		}

		WHEN("The numbers are divided in reverse") {
			T x[3] = {T{33}, T{55}, T{11}};
			T y[3] = {T{3}, T{-5}, T{-1}};
			SIMD::Divide(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3} / T {33});
				REQUIRE(r[1] == T {-5} / T {55});
				REQUIRE(r[2] == T {-1} / T {11});
			}
		}

		WHEN("The numbers are divided by zero") {
			T x[3] = {T{33}, T{55}, T{11}};
			T y[3] = {T{3}, T{-5}, T{0}};

			THEN("The division should throw") {
				REQUIRE_THROWS(SIMD::Divide(x, y, r));
			}
		}

		WHEN("The numbers are min'd") {
			T x[3] = {T{33}, T{-5}, T{0}};
			T y[3] = {T{3}, T{55}, T{12}};
			SIMD::Min(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3});
				REQUIRE(r[1] == T {-5});
				REQUIRE(r[2] == T {0});
			}
		}

		WHEN("The numbers are min'd in reverse") {
			T x[3] = {T{33}, T{-5}, T{0}};
			T y[3] = {T{3}, T{55}, T{12}};
			SIMD::Min(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {3});
				REQUIRE(r[1] == T {-5});
				REQUIRE(r[2] == T {0});
			}
		}

		WHEN("The numbers are max'd") {
			T x[3] = {T{33}, T{-5}, T{0}};
			T y[3] = {T{3}, T{55}, T{12}};
			SIMD::Max(x, y, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33});
				REQUIRE(r[1] == T {55});
				REQUIRE(r[2] == T {12});
			}
		}

		WHEN("The numbers are max'd in reverse") {
			T x[3] = {T{33}, T{-5}, T{0}};
			T y[3] = {T{3}, T{55}, T{12}};
			SIMD::Max(y, x, r);

			THEN("The result should be correct") {
				REQUIRE(r[0] == T {33});
				REQUIRE(r[1] == T {55});
				REQUIRE(r[2] == T {12});
			}
		}
	}
}