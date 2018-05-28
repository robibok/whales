def cov_matrix(sigma1, sigma2, rho):
        cov = [[sigma1 ** 2, rho * sigma1 * sigma2],
                [rho * sigma1 * sigma2, sigma2 ** 2]]
        return cov

