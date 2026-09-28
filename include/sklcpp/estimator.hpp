#pragma once
#include <Eigen/Dense>


namespace sklcpp {

// Base interface for all estimators
class Estimator {
public:
    virtual ~Estimator() = default;  // virtual destructor
};



// Base interface for regression models
class RegressorBase : public Estimator {
public:
    virtual ~RegressorBase() = default;  // virtual destructor

    virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) = 0;
    virtual Eigen::VectorXd predict(const Eigen::MatrixXd& X) const = 0;

    virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) const {
        // R^2 metric
        Eigen::VectorXd y_pred = predict(X);
        double y_avg = y.mean();
        double ss_res = (y - y_pred).squaredNorm();
        double ss_tot = (y - Eigen::VectorXd::Constant(y.size(), y_avg)).squaredNorm();
        return (ss_tot < 1e-12) ? 1.0 : (1.0 - ss_res / ss_tot);
    }
};



// Base interface for classification models
class ClassifierBase : public Estimator {
public:
    virtual ~ClassifierBase() = default;  // virtual destructor

    virtual void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) = 0;
    virtual Eigen::VectorXi predict(const Eigen::MatrixXd& X) const = 0;
    virtual Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& X) const = 0;

    virtual double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const {
        // Accuracy metric
        Eigen::VectorXi y_pred = predict(X);
        double accuracy = (y_pred.array() == y.array()).cast<double>().mean();
        return accuracy;
    }
};

} // namespace sklcpp
