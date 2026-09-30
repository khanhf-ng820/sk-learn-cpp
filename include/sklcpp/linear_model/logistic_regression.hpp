#pragma once
#include <Eigen/Dense>
#include <sklcpp/estimator.hpp>
#include <cmath>


namespace sklcpp {

double sigmoid(double z) {
	return 1.0 / (1.0 + std::exp(-z));
}

double sigmoid_derivative(double z) {
	return sigmoid(z) * (1.0 - sigmoid(z));
}


class LogisticRegression : public ClassifierBase {
public:
	// enum class Method { GradientDescent };

	explicit LogisticRegression(double lr = .01, int n_iters = 100)
		: lr_(lr), n_iters_(n_iters) {}

	const Eigen::VectorXd& weights() const { return weights_; }
	double bias() const { return bias_; }

	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) override {
		if (X.rows() != y.rows())
			throw std::invalid_argument("[ERROR] X and y must have the same number of rows.");
		if (!(((y.array() == 0) || (y.array() == 1)).all()))
			throw std::invalid_argument("[ERROR] y must have binary elements (either 0 or 1).");

		const auto n_samples = X.rows();
		const auto n_features = X.cols();

		weights_ = Eigen::VectorXd::Zero(n_features);
		bias_ = 0;
		for (int i = 0; i < n_iters_; ++i) {
			Eigen::VectorXd y_pred = X * weights_ + Eigen::VectorXd::Constant(n_samples, bias_);
			y_pred = y_pred.array().unaryExpr([](double x) { return sigmoid(x); });
			Eigen::VectorXd error = y_pred - y.cast<double>();
			Eigen::VectorXd dW = (1.0 / n_samples) * (X.transpose() * error);
			double db = (1.0 / n_samples) * error.sum();
			weights_ -= lr_ * dW;
			bias_ -= lr_ * db;
		}
	}

	Eigen::VectorXi predict(const Eigen::MatrixXd& X) const override {
		return predict_proba(X).array().round().cast<int>(); // threshold = 0.5
	}

	Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const override {
		Eigen::MatrixXd z = X * weights_ + Eigen::VectorXd::Constant(X.rows(), bias_); // column vector
		return z.array().unaryExpr([](double x) { return sigmoid(x); });
	}


private:
	double lr_;
	int n_iters_;
	Eigen::VectorXd weights_ = Eigen::VectorXd::Zero(1);
	double bias_ = 0;
};

} // namespace sklcpp
