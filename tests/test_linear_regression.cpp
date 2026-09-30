#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sklcpp/preprocessing.hpp>
#include <sklcpp/linear_model/linear_regression.hpp>
#include <sklcpp/linear_model/logistic_regression.hpp>
#include <iostream>


TEST_CASE("LinearRegression NormalEquation works correctly", "[linear_regression]") {
    // y = 2*x0 + 3*x1 + 1
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 2.0,
         2.0, 1.0,
         3.0, 4.0,
         4.0, 3.0,
         6.0, 7.0,
         8.0, 9.0;

    auto [X_train, X_test] = sklcpp::train_test_split({}, X);
    std::cout << X_train << std::endl << X_test << std::endl;

    Eigen::VectorXd y(6);
    y << 9.0, 8.0, 19.0, 18.0, 34.0, 44.0;

    sklcpp::LinearRegression model(sklcpp::LinearRegression::Method::NormalEquation);
    model.fit(X, y);

    REQUIRE_THAT(model.bias(), Catch::Matchers::WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(model.weights()(0), Catch::Matchers::WithinAbs(2.0, 1e-4));
    REQUIRE_THAT(model.weights()(1), Catch::Matchers::WithinAbs(3.0, 1e-4));

    double r2 = model.score(X, y);
    REQUIRE_THAT(r2, Catch::Matchers::WithinAbs(1.0, 1e-4));
}
