#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sklcpp/linear_model/linear_regression.hpp>


TEST_CASE("LinearRegression NormalEquation works correctly", "[linear_regression]") {
    // y = 2*x0 + 3*x1 + 1
    Eigen::MatrixXd X(4, 2);
    X << 1.0, 2.0,
         2.0, 1.0,
         3.0, 4.0,
         4.0, 3.0;

    Eigen::VectorXd y(4);
    y << 9.0, 8.0, 19.0, 18.0;

    sklcpp::LinearRegression model(sklcpp::LinearRegression::Method::NormalEquation);
    model.fit(X, y);

    REQUIRE_THAT(model.bias(), Catch::Matchers::WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(model.weights()(0), Catch::Matchers::WithinAbs(2.0, 1e-4));
    REQUIRE_THAT(model.weights()(1), Catch::Matchers::WithinAbs(3.0, 1e-4));

    double r2 = model.score(X, y);
    REQUIRE_THAT(r2, Catch::Matchers::WithinAbs(1.0, 1e-4));
}
