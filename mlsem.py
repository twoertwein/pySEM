# standard errors and model fit
# Bayesian?
import warnings
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import torch
from typeguard import typechecked

import sem


@typechecked
def block_diag(matrices: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Block diagonal from a list of square matrices that have different shapes.
    https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
    """
    ends = torch.LongTensor([m.shape[0] for m in matrices]).cumsum(0)

    block = torch.zeros(
        (ends[-1], ends[-1]), device=matrices[0].device, dtype=matrices[0].dtype
    )

    start = 0
    for index, (matrix, end) in enumerate(zip(matrices, ends)):
        if index != 0:
            start = ends[index - 1]
        block[start:end, start:end] = matrix
    return block


class MLSEM(sem.SEM):
    @typechecked
    def __init__(
        self,
        data: pd.DataFrame = None,
        cluster: str = "cluster",
        observed: Sequence[str] = (),
        observed_l2: Sequence[str] = (),
        latent_l2: Sequence[str] = (),
        lambda_y_l2: Optional[Mapping[Tuple[str, str], float]] = None,
        beta_l2: Optional[MutableMapping[Tuple[str, str], float]] = None,
        psi_l2: Optional[MutableMapping[Tuple[str, str], float]] = None,
        theta_l2: Optional[MutableMapping[Tuple[str, str], float]] = None,
        nu_l2: Optional[MutableMapping[str, float]] = None,
        alpha_l2: Optional[MutableMapping[str, float]] = None,
        biased_cov: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Two-level SEM solved with gradient-descend (according to "Multilevel
        structural equation modeling", du Toit et al., 2008).

        Same arguments as SEM with the following additions/changes:

        observed        Name of observed variables at level 1.
        latent          Name of latent variables at level 1.
        observed_l2     Name of observed variables at level 2.
        latent_l2       Name of latent variables at level 2.
        data            The data to fit.
        lambda_y_l2     Loading coefficients (observed x latent; default: zero),
                        from latent to observed.
        beta_l2         Regression between latent variables (latent x latent;
                        default zero), from latent2 to latent1.
        psi_l2          (Co)Variance of latent variables. (latent x latent;
                        default: identity)
        theta_l2        (Co)Variance of residuals (observed x observed; default: diagonal).
        nu              Means of the observed variables at level 1(default: zero).
        nu_l2           Means of the observed variables (default: learnable).
        alpha_l2        Means of the latent variables (default: zero).
        """
        # default nu to zeros
        kwargs.setdefault("nu", {})
        kwargs["nu"] = {name: kwargs["nu"].get(name, 0.0) for name in observed}

        observed = sorted(observed)
        super().__init__(data=data, observed=observed, biased_cov=biased_cov, **kwargs)

        # validate input
        assert observed_l2 and latent_l2 and data is not None
        lambda_y_l2, beta_l2, psi_l2, theta_l2, alpha_l2, nu_l2 = sem._default_settings(
            lambda_y_l2,
            beta_l2,
            psi_l2,
            theta_l2,
            alpha_l2,
            nu_l2,
            latent_l2,
            observed_l2,
        )

        self.observed_l2 = sorted(observed_l2)
        self.latent_l2 = sorted(latent_l2)

        n_observed = len(self.observed_l2)
        n_latent = len(self.latent_l2)

        # warn about level1 observed variables not being used in level2
        in_l1_not_in_l2 = [x for x in observed if x not in self.observed_l2]
        self.naive_implementation = bool(in_l1_not_in_l2)
        if self.naive_implementation:
            warnings.warn(f"{in_l1_not_in_l2} are not used in level 2!")

        # mappings
        mapping = {x: i for i, x in enumerate(self.observed_l2)}
        mapping.update({x: i for i, x in enumerate(self.latent_l2)})

        # save data
        self.names = observed
        self.names_l2 = [x for x in self.observed_l2 if x not in self.names]
        cluster = data[cluster].values
        data = data.loc[:, self.names + self.names_l2].values
        self.missing_patterns, data, cluster = self._init_clusters(
            data=data, clusters=cluster, cluster_vars=len(self.names_l2)
        )
        data = pd.DataFrame(data, columns=self.names + self.names_l2)
        self.register_buffer(
            "data_ys", torch.from_numpy(data.loc[:, self.names].values).double()
        )
        self.register_buffer(
            "data_xs", torch.from_numpy(data.loc[:, self.names_l2].values).double()
        )

        # parameters
        self.variables = [
            "lambda_y_l2",
            "beta_l2",
            "psi_l2",
            "theta_l2",
            "alpha_l2",
            "nu_l2",
        ]
        for name, parameter in sem._get_initialized_parameter(
            data.loc[:, self.observed_l2].groupby(cluster).mean(),
            n_observed,
            n_latent,
            biased=biased_cov,
        ).items():
            name += "_l2"
            setattr(self, name, parameter)

        # and fixed entries
        locals_ = locals()
        self._init_fixed(
            mapping, **{name: locals_[name] for name in self.variables}
        )
        self.variables = [x[:-3] for x in self.variables] + self.variables

        # necessary to invert beta?
        self.cfa_l2 = self.free_parameters["beta_l2"] == 0

        # prepare selection matrices
        self._init_selection_matrices()

    @typechecked
    def _init_clusters(
        self,
        data: Optional[np.ndarray] = None,
        clusters: Optional[np.ndarray] = None,
        cluster_vars: int = -1,
    ) -> Tuple[List[Tuple[slice, List[slice]]], np.ndarray, np.ndarray]:
        # find all missing patterns within clusters and group by the number of
        # missing patterns (for batched inverse and logdet)
        missing_patterns = {}

        # sort by missingness to reduce number of unique S_js
        isnan = np.isnan(data[:, :-cluster_vars])
        index = np.argsort((np.power(2, np.arange(isnan.shape[1])) * isnan).sum(axis=1))
        data = data[index, :]
        clusters = clusters[index]

        data_ = np.zeros_like(data)
        clusters_ = np.zeros_like(clusters)
        start_cluster = 0
        isnan = np.isnan(data[:, :-cluster_vars])
        for cluster in np.unique(clusters):
            missing_patterns[cluster] = []
            cluster_index = cluster == clusters

            subject_index = np.ones(cluster_index.sum(), dtype=np.bool)
            isnan_ = isnan[cluster_index, :]
            start = start_cluster
            while subject_index.any():
                i = np.where(subject_index)[0][0]
                same_pattern = (isnan_[i, None, :] == isnan_).all(axis=1)
                subject_index = subject_index & ~same_pattern

                # reorder data
                slice_ = slice(start, start + same_pattern.sum())
                data_[slice_, :] = data[cluster_index, :][same_pattern, :]
                clusters_[slice_] = clusters[cluster_index][same_pattern]
                missing_patterns[cluster].append(slice_)
                start += same_pattern.sum()
            # add slices
            missing_patterns[cluster] = (
                slice(start_cluster, start),
                missing_patterns[cluster],
            )
            start_cluster = start
        return list(missing_patterns.values()), data_, clusters_

    @typechecked
    def _init_selection_matrices(self) -> None:
        # transform between sigma/mu
        pure_l2 = torch.eye(len(self.observed_l2), dtype=torch.double)[
            torch.BoolTensor([x in self.names_l2 for x in self.observed_l2]), :
        ]
        pure_l1 = torch.zeros(
            (len(self.observed), len(self.observed_l2)), dtype=torch.double
        )
        for i, l1 in enumerate(self.observed):
            j = [j for j, l2 in enumerate(self.observed_l2) if l1 == l2]
            if not j:
                continue
            pure_l1[i, j[0]] = 1.0
        self.register_buffer("pure_l1", pure_l1)
        self.register_buffer("pure_l2", pure_l2)

    @typechecked
    def _split(
        self, sigma: torch.Tensor, mu: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decomposes the between covariances/means into:
        - sigma_b: level-1 variables only
        - sigma_xx: level-2 variables only
        - sigma_yx: cov(level-1, level-2)
        - mu_w and mu_b: same dimensions, level-1 variables only
        - mu_x: level-2 variables only

        Returns:
            sigma_b, sigma_xx, sigma_yx, mu_b, mu_x

        """
        sigma_xx = self.pure_l2.mm(sigma.mm(self.pure_l2.t()))
        if self.pure_l2.shape[0] == 0:
            mu_x = torch.empty((0, 1), device=sigma.device, dtype=sigma.dtype)
        else:
            mu_x = self.pure_l2.mm(mu)

        sigma_b = self.pure_l1.mm(sigma.mm(self.pure_l1.t()))
        mu_b = self.pure_l1.mm(mu)

        sigma_yx = self.pure_l1.mm(sigma.mm(self.pure_l2.t()))

        return (sigma_b, sigma_xx, sigma_yx, mu_b, mu_x)

    @typechecked
    def forward(self) -> Tuple[torch.Tensor, int]:
        unstable = 0

        # get sigma/mean or each level
        sigma_w, mu_w = self.implied_sigma_mu()
        sigma_l2, mu_l2 = self.implied_sigma_mu(suffix="_l2")

        # decompose into 11, 12, 21, 22
        sigma_b, sigma_xx, sigma_yx, mu_b, mu_x = self._split(sigma_l2, mu_l2)

        # cluster FIML -2 * logL (without constants)
        loss = torch.zeros(1, dtype=mu_w.dtype, device=mu_w.device)

        # go through each cluster separately
        data_ys_available = ~torch.isnan(self.data_ys)
        cache_S_ij = {}
        cache_S_j_R_j = {}
        sigma_b_logdet = None
        sigma_b_inv = None
        if not self.naive_implementation:
            sigma_b_logdet = torch.logdet(sigma_b)
            sigma_b_inv = torch.inverse(sigma_b)
        for cluster_slice, batches in self.missing_patterns:
            # get cluster data and define R_j for current cluster j
            cluster_x = self.data_xs[cluster_slice.start, :]
            R_j_index = ~torch.isnan(cluster_x)
            no_cluster = ~R_j_index.any()

            # cache
            key = (
                tuple(R_j_index.tolist()),
                tuple([tuple(x) for x in data_ys_available[cluster_slice, :].tolist()]),
            )
            sigma_j_logdet, sigma_j_inv = cache_S_j_R_j.get(key, (None, None))

            # define S_ij and S_j
            S_ijs = []
            eye_w = torch.eye(mu_w.shape[0], dtype=mu_w.dtype, device=mu_w.device)
            lambda_ijs_logdet_sum = 0.0
            lambda_ijs_inv = []
            A_j = torch.zeros_like(sigma_w)
            for batch_slice in batches:
                size = batch_slice.stop - batch_slice.start
                available = data_ys_available[batch_slice.start]
                S_ij = eye_w[available, :]
                S_ijs.extend([S_ij] * size)

                if self.naive_implementation or sigma_j_logdet is not None:
                    continue

                key_S_ij = tuple(available.tolist())
                lambda_ij_inv, lambda_ij_logdet, a_j = cache_S_ij.get(
                    key_S_ij, (None, None, None)
                )

                if lambda_ij_inv is None:
                    lambda_ij = sigma_w  # no missing data
                    if S_ij.shape[0] != eye_w.shape[0]:
                        # missing data
                        lambda_ij = S_ij.mm(sigma_w.mm(S_ij.t()))
                    lambda_ij_inv = torch.inverse(lambda_ij)
                    lambda_ij_logdet = torch.logdet(lambda_ij)

                    if S_ij.shape[0] != eye_w.shape[0]:
                        # missing data
                        a_j = S_ij.t().mm(lambda_ij_inv.mm(S_ij))
                    else:
                        a_j = lambda_ij_inv
                    cache_S_ij[key_S_ij] = lambda_ij_inv, lambda_ij_logdet, a_j

                lambda_ijs_inv.extend([lambda_ij_inv] * size)
                lambda_ijs_logdet_sum = lambda_ijs_logdet_sum + lambda_ij_logdet * size
                A_j = A_j + a_j * size

            S_j = torch.cat(S_ijs, dim=0)

            # means
            y_j = torch.cat(
                [
                    self.data_ys[cluster_slice, :][data_ys_available[cluster_slice, :]][
                        :, None
                    ],
                    cluster_x[R_j_index, None],
                ]
            )
            mu_y = mu_w + mu_b
            mu_j = torch.cat([S_j.mm(mu_y), mu_x[R_j_index]])
            mean_diff = y_j - mu_j
            G_yj = mean_diff.mm(mean_diff.t())

            if sigma_j_logdet is None and not self.naive_implementation:
                sigma_b_inv_A_j = sigma_b_inv + A_j
                B_j = torch.inverse(sigma_b_inv_A_j)
                C_j = eye_w - A_j.mm(B_j)
                D_j = C_j.mm(A_j)
                lambda_inv = block_diag(lambda_ijs_inv)
                V_j_inv = lambda_inv - lambda_inv.mm(
                    S_j.mm(B_j.mm(S_j.t().mm(lambda_inv)))
                )

                if no_cluster:
                    # no cluster
                    sigma_11_j = V_j_inv
                    sigma_21_j = torch.empty(
                        0, device=sigma_11_j.device, dtype=sigma_11_j.dtype
                    )
                    sigma_22_1 = torch.empty(
                        [0, 0], device=sigma_11_j.device, dtype=sigma_11_j.dtype
                    )
                    sigma_22_inv = sigma_21_j

                else:
                    # normal case
                    sigma_22_1 = (sigma_xx - sigma_yx.t().mm(D_j.mm(sigma_yx)))[
                        R_j_index, :
                    ][:, R_j_index]
                    sigma_22_inv = torch.inverse(sigma_22_1)
                    sigma_jyx = S_j.mm(sigma_yx[:, R_j_index])
                    sigma_11_j = (
                        V_j_inv.mm(
                            sigma_jyx.mm(sigma_22_inv.mm(sigma_jyx.t().mm(V_j_inv)))
                        )
                        + V_j_inv
                    )
                    sigma_21_j = -sigma_22_inv.mm(sigma_jyx.t().mm(V_j_inv))

                sigma_j_inv = torch.cat(
                    [
                        torch.cat([sigma_11_j, sigma_21_j]),
                        torch.cat([sigma_21_j.t(), sigma_22_inv]),
                    ],
                    dim=1,
                )

                sigma_j_logdet = (
                    lambda_ijs_logdet_sum
                    + sigma_b_logdet
                    + torch.logdet(sigma_b_inv_A_j)
                    + torch.logdet(sigma_22_1)
                )
                cache_S_j_R_j[key] = (sigma_j_logdet, sigma_j_inv)

            elif sigma_j_logdet is None:
                # naive
                sigma_j = S_j.mm(sigma_b.mm(S_j.t())) + block_diag(
                    [S_ij.mm(sigma_w.mm(S_ij.t())) for S_ij in S_ijs]
                )
                if not no_cluster:
                    sigma_j_12 = S_j.mm(sigma_yx[:, R_j_index])
                    sigma_j_21 = sigma_j_12.t()
                    sigma_j_22 = sigma_xx[R_j_index, :][:, R_j_index]
                    sigma_j = torch.cat(
                        [
                            torch.cat([sigma_j, sigma_j_21]),
                            torch.cat([sigma_j_12, sigma_j_22]),
                        ],
                        dim=1,
                    )
                sigma_j_logdet = torch.logdet(sigma_j)
                sigma_j_inv = torch.inverse(sigma_j)
                cache_S_j_R_j[key] = (sigma_j_logdet, sigma_j_inv)

            loss_current = sigma_j_logdet + torch.trace(sigma_j_inv.mm(G_yj))
            unstable += loss_current.detach().item() < 0
            loss = loss + loss_current.clamp(min=0.0)

        return loss, unstable
