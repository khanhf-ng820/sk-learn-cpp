#pragma once
#include <sklcpp/estimator.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>


namespace sklcpp {

// Preprocessing utilities: StandardScaler

class StandardScaler : public Estimator {
public:
	explicit StandardScaler(bool with_mean = true, bool with_std = true, bool copy = true)
		: with_mean_(with_mean), with_std_(with_std), copy_(copy) {}

	const Eigen::RowVectorXd& mean() const { return mean_; }
	const Eigen::RowVectorXd& stddev() const { return stddev_; }

	void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) { // y is ignored
		mean_ = X.colwise().mean();
		Eigen::RowVectorXd sum_sq = (X.rowwise() - mean_).array().square().colwise().sum();
		stddev_ = (sum_sq / X.rows()).array().sqrt();
		stddev_ = stddev_.array().unaryExpr([](double x) { return x = (x < 1e-12 ? 1. : x); });
	}

	std::optional<Eigen::MatrixXd> transform(Eigen::MatrixXd& X) const {
		assert((stddev_.array() > 0.).all() && X.cols() == mean_.size() && X.cols() == stddev_.size());

		if (copy_) {
			Eigen::MatrixXd X_copy = X;
			X_copy = (X_copy.array().rowwise() - mean_.array()).rowwise() / stddev_.array();
			return X_copy;
		} else {
			X = (X.array().rowwise() - mean_.array()).rowwise() / stddev_.array();
			return std::nullopt;
		}
	}

	std::optional<Eigen::MatrixXd> fit_transform(Eigen::MatrixXd& X) {
		mean_ = X.colwise().mean();
		Eigen::RowVectorXd sum_sq = (X.rowwise() - mean_).array().square().colwise().sum();
		stddev_ = (sum_sq / X.rows()).array().sqrt();

		if (copy_) {
			Eigen::MatrixXd X_copy = X;
			X_copy = (X_copy.array().rowwise() - mean_.array()).rowwise() / stddev_.array();
			return X_copy;
		} else {
			X = (X.array().rowwise() - mean_.array()).rowwise() / stddev_.array();
			return std::nullopt;
		}
	}

	std::optional<Eigen::MatrixXd> inverse_transform(Eigen::MatrixXd& X) {
		assert((stddev_.array() >= 0.).all() && X.cols() == mean_.size() && X.cols() == stddev_.size());

		if (copy_) {
			Eigen::MatrixXd X_copy = X;
			X_copy = X_copy.array().rowwise() * stddev_.array() + mean_.array();
			return X_copy;
		} else {
			X = X.array().rowwise() * stddev_.array() + mean_.array();
			return std::nullopt;
		}
	}

private:
	bool with_mean_ = true; // centers the data before scaling
	bool with_std_ = true; // scales the data to unit standard deviation
	bool copy_ = true; // if false, performs in-place
	Eigen::RowVectorXd mean_;
	Eigen::RowVectorXd stddev_;
};





// Preprocessing utilities: train_test_split

struct SplitOptions {
	double test_size = 0.25; // in range (0.0, 1.0)
	bool shuffle = true;
	std::optional<std::uint64_t> seed = std::nullopt;
};

namespace detail {

	struct SplitIndices {
		std::vector<Eigen::Index> train;
		std::vector<Eigen::Index> test;
	};

	inline SplitIndices make_split_indices(Eigen::Index n_samples, const SplitOptions& opts) {
		if (!(opts.test_size > 0.0 && opts.test_size < 1.0)) {
			throw std::invalid_argument("[ERROR] test_size must be strictly between 0 and 1.");
		}

		// Same rule as sklearn: n_test = ceil(test_size * n), n_train = the rest
		const auto n_test = static_cast<Eigen::Index>(std::ceil(opts.test_size * static_cast<double>(n_samples)));
		const Eigen::Index n_train = n_samples - n_test;
		if (n_train < 1 || n_test < 1) {
			throw std::invalid_argument("[ERROR] Too few samples for this test_size: both splits must be non-empty.");
		}

		std::vector<Eigen::Index> idx(static_cast<std::size_t>(n_samples));
		std::iota(idx.begin(), idx.end(), Eigen::Index{0});

		if (opts.shuffle) {
			std::mt19937_64 rng(opts.seed ? *opts.seed : std::random_device{}());
			std::shuffle(idx.begin(), idx.end(), rng);
		}

		SplitIndices out;
		out.train.assign(idx.begin(), idx.begin() + n_train);
		out.test.assign(idx.begin() + n_train, idx.end());
		return out;
	}

	// Splits ONE array using precomputed indices -> tuple<train, test>
	template <typename Derived>
	std::tuple<typename Derived::PlainObject, typename Derived::PlainObject> split_one(const Eigen::MatrixBase<Derived>& a, const SplitIndices& s) {
		static_assert(Derived::RowsAtCompileTime == Eigen::Dynamic,
					  "train_test_split needs dynamic row counts, with samples as rows (use MatrixXd/VectorXd, not RowVectorXd or fixed-size types).");
		using Plain = typename Derived::PlainObject;

		Plain train = a(s.train, Eigen::all); // explicit type: forces evaluation
		Plain test  = a(s.test,  Eigen::all);
		return std::tuple<Plain, Plain>(std::move(train), std::move(test));
	}

}  // namespace detail

template <typename... Derived>
auto train_test_split(const SplitOptions& opts, const Eigen::MatrixBase<Derived>&... arrays) {
	static_assert(sizeof...(Derived) > 0, "Pass at least one array to split.");

	// All arrays must agree on the number of samples
	const std::array<Eigen::Index, sizeof...(Derived)> rows{arrays.rows()...};
	for (const auto r : rows) {
		if (r != rows[0]) {
			throw std::invalid_argument("All arrays passed to train_test_split must have the same number of rows.");
		}
	}
	
	// ONE shared index set, applied identically to every array
	const auto split = detail::make_split_indices(rows[0], opts);

	// {X_train, X_test} + {y_train, y_test} + ... -> flat tuple
	return std::tuple_cat(detail::split_one(arrays, split)...);
}

// Convenience overload: default options.  train_test_split(X, y)
template <typename... Derived>
auto train_test_split(const Eigen::MatrixBase<Derived>&... arrays) {
	return train_test_split(SplitOptions{}, arrays...);
}


} // namespace sklcpp
