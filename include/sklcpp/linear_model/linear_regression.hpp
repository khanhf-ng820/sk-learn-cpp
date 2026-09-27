#pragma once
#include <Eigen/Dense>
#include <stdexcept>

namespace sklcpp {

class LinearRegression {
public:
	enum class Method { NormalEquation, GradientDescent };

	explicit LinearRegression(Method method = Method::NormalEquation, double lr = .01, int n_iters = 100)
		: method_(method), lr_(lr), n_iters_(n_iters) {}

	const Eigen::VectorXd& weights() const { return weights_; }
	double bias() const { return bias_; }

	LinearRegression& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
		if (X.rows() != y.rows())
			throw std::invalid_argument("[ERROR] X and y must have the same number of rows.");

		const auto n_samples = X.rows();
		const auto n_features = X.cols();

		if (method_ == Method::NormalEquation) {
			Eigen::MatrixXd X_b(n_samples, n_features + 1);
			X_b.col(0).setOnes();
			X_b.rightCols(n_features) = X;

			Eigen::VectorXd x_hat = (X_b.transpose() * X_b).ldlt().solve(X_b.transpose() * y);
			bias_ = x_hat(0);
			weights_ = x_hat.tail(n_features);
		} else {
			weights_ = Eigen::VectorXd::Zero(n_features);
			bias_ = 0;
			for (int i = 0; i < n_iters_; ++i) {
				Eigen::VectorXd y_pred = X * weights_ + Eigen::VectorXd::Constant(n_samples, bias_);
				Eigen::VectorXd error = y_pred - y;
				Eigen::VectorXd dW = (1.0 / n_samples) * (X.transpose() * error);
				double db = (1.0 / n_samples) * error.sum();
				weights_ -= lr_ * dW;
				bias_ -= lr_ * db;
			}
		}

		return *this;
	}

	Eigen::VectorXd predict(const Eigen::MatrixXd& X) const {
		return X * weights_ + Eigen::VectorXd::Constant(X.rows(), bias_);
	}

	double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
		Eigen::VectorXd y_pred = predict(X);
		double y_avg = y.mean();
		double ss_res = (y - y_pred).squaredNorm();
		double ss_tot = (y - Eigen::VectorXd::Constant(y.size(), y_avg)).squaredNorm();
		return 1.0 - ss_res / ss_tot;
	}


private:
	Method method_;
	double lr_;
	int n_iters_;
	Eigen::VectorXd weights_ = Eigen::VectorXd::Zero(1);
	double bias_ = 0.;
};

} // namespace sklcpp
