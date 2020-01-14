import warnings
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from typeguard import typechecked


@typechecked
def _convert_indices(
    dictionary: Mapping[Union[Tuple[str, str], str], float],
    shape: Tuple[int, int],
    mapping: Mapping[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a mask and matrix from a dictionary."""
    mask = torch.ones(shape, dtype=torch.bool)
    fixed = torch.zeros(shape, dtype=torch.double)
    for key, constant in dictionary.items():
        if isinstance(key, tuple):
            key = mapping[key[0]], mapping[key[1]]
        else:
            key = mapping[key]
        if not constant == constant:
            mask[key] = False
        else:
            fixed[key] = constant
    return mask, fixed


@typechecked
def _symmetric_dict(
    dictionary: MutableMapping[Tuple[str, str], float]
) -> Mapping[Tuple[str, str], float]:
    """Make a dictionary symmetric."""
    for (i, j), constant in list(dictionary.items()):
        dictionary[j, i] = constant
    return dictionary


@typechecked
def _exclusive_dicts(
    dict_a: Mapping[Tuple[str, str], float],
    dict_b: MutableMapping[Tuple[str, str], float],
) -> Mapping[Tuple[str, str], float]:
    """Non-zero in A -> 0 in B."""
    for key, value in dict_a.items():
        if value == 0:
            continue
        dict_b[key] = 0
    return dict_b


@typechecked
def _default_settings(
    lambda_y: Optional[Mapping[Tuple[str, str], float]],
    beta: Optional[Mapping[Tuple[str, str], float]],
    psi: Optional[Mapping[Tuple[str, str], float]],
    theta: Optional[Mapping[Tuple[str, str], float]],
    alpha: Optional[Mapping[str, float]],
    nu: Optional[Mapping[str, float]],
    latent: Sequence[str],
    observed: Sequence[str],
) -> (
    Mapping[Tuple[str, str], float],
    Mapping[Tuple[str, str], float],
    Mapping[Tuple[str, str], float],
    Mapping[Tuple[str, str], float],
    Mapping[str, float],
    Mapping[str, float],
):
    if lambda_y is None:
        lambda_y = {}
    if beta is None:
        beta = {}
    if psi is None:
        psi = {}
    if theta is None:
        theta = {}
    if alpha is None:
        alpha = {name: 0.0 for name in latent}
    if nu is None:
        nu = {}

    theta = _symmetric_dict(theta)
    psi = _exclusive_dicts(beta, psi)
    beta = _exclusive_dicts(psi, beta)
    psi = _symmetric_dict(psi)

    # defaults
    for name in latent:
        psi.setdefault((name, name), 1.0)
    for name in observed:
        theta.setdefault((name, name), float("NaN"))
        nu.setdefault(name, float("NaN"))

    return lambda_y, beta, psi, theta, alpha, nu


@typechecked
def _get_initialized_parameter(
    data: pd.DataFrame, n_observed: int, n_latent: int, biased: bool = False
) -> Dict[str, torch.nn.Parameter]:
    """Similar to lavaan's simple."""
    scale = 1.0
    if biased:
        scale = (data.shape[0] - 1.0) / data.shape[0]

    return {
        "lambda_y": torch.nn.Parameter(
            torch.ones((n_observed, n_latent), dtype=torch.double)
        ),
        "beta": torch.nn.Parameter(
            torch.zeros((n_latent, n_latent), dtype=torch.double)
        ),
        "psi": torch.nn.Parameter(torch.eye(n_latent, dtype=torch.double)),
        "theta": torch.nn.Parameter(
            torch.from_numpy(
                (data.cov().abs() * scale / 2.0).values * np.eye(n_observed)
            )
            .double()
            .clamp(min=0.1)
        ),
        "alpha": torch.nn.Parameter(torch.zeros((n_latent, 1), dtype=torch.double)),
        "nu": torch.nn.Parameter(
            torch.from_numpy(data.mean().values[:, None]).double()
        ),
    }


class SEM(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        observed: Sequence[str] = (),
        latent: Sequence[str] = (),
        data: Optional[pd.DataFrame] = None,
        lambda_y: Optional[Mapping[Tuple[str, str], float]] = None,
        beta: Optional[MutableMapping[Tuple[str, str], float]] = None,
        psi: Optional[MutableMapping[Tuple[str, str], float]] = None,
        theta: Optional[MutableMapping[Tuple[str, str], float]] = None,
        nu: Optional[MutableMapping[str, float]] = None,
        alpha: Optional[MutableMapping[str, float]] = None,
        biased_cov: bool = True,
    ) -> None:
        """
        A SEM solved with gradient-descend.

        observed        Name of observed variables.
        latent          Name of latent variables.
        data            The data to fit.
        lambda_y/beta/psi/theta     A dictionary [(i, j), c] to enforce a constant
                        c in the matrix at position (i, j). If c is NaN, (i, j)
                        is a learnable parameter.
        lambda_y        Loading coefficients (observed x latent; default: zero),
                        from latent to observed.
        beta            Regression between latent variables (latent x latent;
                        default zero), from latent2 to latent1.
        psi             (Co)Variance of latent variables. (latent x latent;
                        default: identity)
        theta           (Co)Variance of residuals (observed x observed; default: diagonal).
        nu              Means of the observed variables (default: None/learnable).
        alpha           Means of the latent variables (default: None/zero).
        biased_cov      Whether to use the biased covariance estimation (lavaan uses the
                        biased covariance estimation).

        psi/theta are symmetric: sufficient to specify one pair.

        If a psi is non-zero or free, the corresponding beta is set to zero (and
        vice-versa).

        Maximum Likelihood estimation is used by default. If data is missing or means
        are to be estimated (nu or alpha is not None), Full Information
        Maximum Likelihood will be used.
        """
        super().__init__()
        # validate input
        assert observed and latent and data is not None
        self.fiml = alpha is not None or nu is not None or data.isna().any().any()
        lambda_y, beta, psi, theta, alpha, nu = _default_settings(
            lambda_y, beta, psi, theta, alpha, nu, latent, observed
        )

        self.observed = observed
        self.latent = latent

        n_observed = len(observed)
        n_latent = len(latent)

        # mappings
        mapping = {x: i for i, x in enumerate(observed)}
        mapping.update({x: i for i, x in enumerate(latent)})

        # save data and get sample covariance matrix
        data = data.loc[:, observed]
        sample_covariance = data.cov()
        self.register_buffer("data", torch.from_numpy(data.values).double())
        if biased_cov:
            samples = data.shape[0]
            sample_covariance = sample_covariance * (samples - 1.0) / samples
        self.register_buffer(
            "sample_covariance", torch.from_numpy(sample_covariance.values).double()
        )

        # parameters
        self.variables = ["lambda_y", "beta", "psi", "theta"]
        if self.fiml:
            self.variables += ["alpha", "nu"]
        for name, parameter in _get_initialized_parameter(
            data, n_observed, n_latent, biased=biased_cov
        ).items():
            if name not in self.variables:
                continue
            setattr(self, name, parameter)

        # and fixed entries
        self.free_parameters: Dict[str, int] = {}
        locals_ = locals()
        self._init_fixed(mapping, **{name: locals_[name] for name in self.variables})

        # necessary to invert beta?
        self.cfa = self.free_parameters["beta"] == 0

        # find all missingness patterns
        self.missing_patterns: Dict = {}
        if self.fiml:
            self.missing_patterns = self._init_fiml()

    @typechecked
    def _init_fixed(
        self,
        mapping: Mapping[Union[Tuple[str, str], str], float],
        **kwargs: Mapping[Union[str, Tuple[str, str]], float],
    ) -> None:
        for name, fixed in kwargs.items():
            mask, constants = _convert_indices(
                fixed, getattr(self, name).shape, mapping
            )
            self.register_buffer(f"{name}_mask", mask)
            self.register_buffer(f"{name}_fixed", constants)

            # number of free parameters
            if name in ("psi", "theta"):
                free = torch.triu(~mask).sum()
            else:
                free = (~mask).sum()
            self.free_parameters[name] = int(free.item())

    @typechecked
    def _init_fiml(self) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # find all missing patterns and group by the number of missing patterns (for
        # batched inverse and logdet)
        data = self.data
        missing_patterns: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        index = torch.ones(data.shape[0], dtype=torch.bool)
        isnan = torch.isnan(data)
        while index.any():
            i = torch.nonzero(index)[0, 0]
            same_pattern = (isnan[i, None, :] == isnan).all(dim=1)
            features = int((~isnan[i, :]).sum().cpu().item())
            if features not in missing_patterns:
                missing_patterns[features] = []
            missing_patterns[features].append(
                (
                    torch.nonzero(same_pattern).flatten(),
                    torch.nonzero(~isnan[i, :]).flatten(),
                )
            )
            index = index & ~same_pattern
        return missing_patterns

    @typechecked
    def get(self, name: str) -> torch.Tensor:
        """
        Return the requested matrix.
        """
        assert name in self.variables

        # mix of free and fixed variables
        matrix = getattr(self, name)
        fixed = getattr(self, f"{name}_fixed")
        mask = getattr(self, f"{name}_mask")
        matrix = torch.where(mask, fixed, matrix)

        # positive
        if name.startswith("theta"):
            matrix = matrix.abs()

        # symmetric variables
        if name.startswith("psi") or name.startswith("theta"):
            matrix = (matrix + matrix.t()) / 2

        return matrix

    @typechecked
    def summary(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Print and return a summary of the estimated parameters.

        verbose     Whether to print the summary.

        Returns:
            Dictionary with estimations as values.
        """
        results = {}
        for name in self.variables:
            index = self.observed
            columns = self.latent
            if name.endswith("_l2"):
                index = self.observed_l2
                columns = self.latent_l2

            if name.startswith("theta"):
                columns = index
            elif name.startswith("beta") or name.startswith("psi"):
                index = columns
            elif name.startswith("nu"):
                columns = None
            elif name.startswith("alpha"):
                index = columns
                columns = None

            results[name] = pd.DataFrame(
                self.get(name).detach().cpu().numpy(), columns=columns, index=index
            )

        # print
        if verbose:
            for name, value in results.items():
                print(name, f"({self.free_parameters[name]} free parameter(s))")
                print(value)
                print()

        return results

    @typechecked
    def fit(self, epochs: int = 20000) -> List[float]:
        """
        Fit the SEM iteratively.

        epochs      Maximal number of epochs to train for.

        Returns:
            List of losses.

        """
        optimizer = torch.optim.Adam(self.parameters())
        losses = []
        best_loss = np.Inf
        best_loss_counter = 0
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss, unstable = self.forward()
            losses.append(loss.detach().item())
            assert losses[-1] >= 0, f"Loss {loss} in epoch {epoch}"
            loss.backward()
            optimizer.step()

            if unstable > 0:
                warnings.warn(
                    f"{unstable} unstable/singular operation(s) during epoch {epoch} ({losses[-1]})"
                )
                if losses[-1] == 0:
                    break
                continue

            # early stopping (best loss has been reached 5 times)
            loss = np.round(losses[-1], 10)
            if loss < best_loss:
                best_loss_counter = 0
                best_loss = loss
            elif loss == best_loss:
                best_loss_counter += 1

            if best_loss_counter >= 5:
                break

        return losses

    @typechecked
    def implied_sigma_mu(
        self, suffix: str = ""
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Calculate the model-implied covariance and mean."""
        psi = self.get(f"psi{suffix}")
        lambda_y = self.get(f"lambda_y{suffix}")
        theta = self.get(f"theta{suffix}")

        if not getattr(self, f"cfa{suffix}"):
            eye = torch.eye(psi.shape[0], dtype=psi.dtype, device=psi.device)
            beta = torch.inverse(eye - self.get(f"beta{suffix}"))
            psi = beta.mm(psi.mm(beta.t()))

        # model-implied covariance
        sigma = lambda_y.mm(psi.mm(lambda_y.t())) + theta
        sigma = (sigma + sigma.t()) / 2

        # mode-implied mean
        mu = None
        if self.fiml:
            alpha = self.get(f"alpha{suffix}")
            if not getattr(self, f"cfa{suffix}"):
                lambda_y = lambda_y.mm(beta)
            mu = self.get(f"nu{suffix}") + lambda_y.mm(alpha)

        return sigma, mu

    @typechecked
    def forward(self) -> Tuple[torch.Tensor, int]:
        """Return a value proportional to the log likelihood."""
        unstable = 0

        # model-implied covariance/mean
        sigma, mu = self.implied_sigma_mu()

        if self.fiml:
            # FIML -2 * logL (without constants)
            assert mu is not None
            loss = torch.zeros(1, dtype=mu.dtype, device=mu.device)
            mean_diffs: torch.Tensor = self.data - mu.t()
            for pairs in self.missing_patterns.values():
                # calculate sigma^-1 and logdet(sigma) in batches
                sigmas = torch.stack(
                    [sigma.index_select(0, x[1]).index_select(1, x[1]) for x in pairs]
                )
                sigmas_logdet = torch.logdet(sigmas)
                sigmas = torch.inverse(sigmas)

                for i, (observations, available) in enumerate(pairs):
                    mean_diff = mean_diffs.index_select(0, observations).index_select(
                        1, available
                    )
                    loss_current = sigmas_logdet[i] * len(observations) + torch.trace(
                        mean_diff.mm(sigmas[i]).mm(mean_diff.t())
                    )
                    unstable += loss_current.detach().item() < 0
                    loss = loss + loss_current.clamp(min=0.0)

        else:
            # maximum likelihood
            loss = torch.logdet(sigma) + torch.trace(
                self.sample_covariance.mm(torch.inverse(sigma))
            )
            unstable += loss.detach().item() < 0

        return loss, unstable
