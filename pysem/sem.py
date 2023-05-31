import warnings
import numpy as np
import pandas as pd
import torch


def _convert_indices(
    dictionary: dict[tuple[str, str] | str, float],
    shape: tuple[int, int],
    mapping: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _symmetric_dict(
    dictionary: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Make a dictionary symmetric."""
    for (i, j), constant in list(dictionary.items()):
        dictionary[j, i] = constant
    return dictionary


def _exclusive_dicts(
    dict_a: dict[tuple[str, str], float],
    dict_b: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    """Non-zero in A -> 0 in B."""
    for key, value in dict_a.items():
        if value == 0:
            continue
        dict_b[key] = 0
    return dict_b


def _default_settings(
    lambda_y: dict[tuple[str, str], float] | None,
    beta: dict[tuple[str, str], float] | None,
    psi: dict[tuple[str, str], float] | None,
    theta: dict[tuple[str, str], float] | None,
    alpha: dict[str, float] | None,
    nu: dict[str, float] | None,
    latent: list[str],
    observed: list[str],
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
    dict[str, float],
    dict[str, float],
]:
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


def _get_initialized_parameter(
    data: pd.DataFrame, n_observed: int, n_latent: int, biased: bool = False,
) -> dict[str, torch.nn.Parameter]:
    """Similar to lavaan's simple."""
    scale = 1.0
    if biased:
        scale = (data.shape[0] - 1.0) / data.shape[0]

    return {
        "lambda_y": torch.nn.Parameter(
            torch.ones(n_observed, n_latent, dtype=torch.double),
        ),
        "beta": torch.nn.Parameter(torch.zeros(n_latent, n_latent, dtype=torch.double)),
        "psi": torch.nn.Parameter(torch.eye(n_latent, dtype=torch.double)),
        "theta": torch.nn.Parameter(
            torch.from_numpy(
                (data.cov().abs() * scale / 2.0).values * np.eye(n_observed),
            )
            .double()
            .clamp(min=0.1),
        ),
        "alpha": torch.nn.Parameter(torch.zeros(n_latent, 1, dtype=torch.double)),
        "nu": torch.nn.Parameter(
            torch.from_numpy(data.mean().values[:, None]).double(),
        ),
    }


class SEM(torch.nn.Module):
    def __init__(
        self,
        *,
        observed: list[str],
        latent: list[str],
        data: pd.DataFrame,
        lambda_y: dict[tuple[str, str], float] | None = None,
        beta: dict[tuple[str, str], float] | None = None,
        psi: dict[tuple[str, str], float] | None = None,
        theta: dict[tuple[str, str], float] | None = None,
        nu: dict[str, float] | None = None,
        alpha: dict[str, float] | None = None,
        biased_cov: bool = True,
    ) -> None:
        """A SEM solved with gradient-descend.

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
        self.fiml = alpha is not None or nu is not None or data.isna().any().any()
        lambda_y, beta, psi, theta, alpha, nu = _default_settings(
            lambda_y, beta, psi, theta, alpha, nu, latent, observed,
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
            "sample_covariance", torch.from_numpy(sample_covariance.values).double(),
        )

        # parameters
        self.variables = ["lambda_y", "beta", "psi", "theta"]
        if self.fiml:
            self.variables += ["alpha", "nu"]
        for name, parameter in _get_initialized_parameter(
            data, n_observed, n_latent, biased=biased_cov,
        ).items():
            if name not in self.variables:
                continue
            setattr(self, name, parameter)

        # and fixed entries
        self.free_parameters: dict[str, int] = {}
        locals_ = locals()
        self._init_fixed(mapping, **{name: locals_[name] for name in self.variables})

        # necessary to invert beta?
        self.cfa = self.free_parameters["beta"] == 0

        # find all missingness patterns
        self.missing_patterns: dict = {}
        if self.fiml:
            self.missing_patterns = self._init_fiml()

    def _init_fixed(
        self,
        mapping: dict[str, int],
        **kwargs: dict[str | tuple[str, str], float],
    ) -> None:
        for name, fixed in kwargs.items():
            # remove duplicated parameters in symmetric matrices
            if name in ("psi", "theta"):
                for (name_a, name_b), value in fixed.items():
                    if value == value:
                        continue
                    rank_a = mapping[name_a]
                    rank_b = mapping[name_b]
                    if rank_a <= rank_b:
                        continue
                    # set coefficient to zero
                    fixed[name_a, name_b] = 0.0
                    # and adjust default value
                    getattr(self, name).data[rank_b, rank_a] *= 2

            mask, constants = _convert_indices(
                fixed, getattr(self, name).shape, mapping,
            )
            self.register_buffer(f"{name}_mask", mask)
            self.register_buffer(f"{name}_fixed", constants)

            # number of free parameters
            free = (~mask).sum()
            self.free_parameters[name] = int(free.item())

    def _init_fiml(self) -> dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        # find all missing patterns and group by the number of missing patterns (for
        # batched inverse and logdet)
        data = self.data
        missing_patterns: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
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
                ),
            )
            index = index & ~same_pattern
        return missing_patterns

    def get(self, name: str) -> torch.Tensor:
        """Return the requested matrix."""
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
        if name.startswith(("psi", "theta")):
            matrix = (matrix + matrix.t()) / 2

        # positive diagonal
        if name.startswith("psi"):
            eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
            matrix = matrix * (1 - eye) + matrix.abs() * eye

        return matrix

    def summary(self, verbose: bool = True) -> dict[str, pd.DataFrame]:
        """Print and return a summary of the estimated parameters.

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
            elif name.startswith(("beta", "psi")):
                index = columns
            elif name.startswith("nu"):
                columns = None
            elif name.startswith("alpha"):
                index = columns
                columns = None

            results[name] = pd.DataFrame(
                self.get(name).detach().cpu().numpy(), columns=columns, index=index,
            )

        if verbose:
            for name, value in results.items():
                print(name, f"({self.free_parameters[name]} free parameter(s))")
                print(value)
                print()

        return results

    def fit(self, epochs: int = 15_000, lr: float = 0.002) -> list[float]:
        """Fit the SEM iteratively.

        epochs      Maximal number of epochs to train for.

        Returns:
            List of losses.

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        best_loss = np.Inf
        best_loss_counter = 0
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = self.forward()
            losses.append(loss.detach().item())
            loss.backward()
            optimizer.step()

            # early stopping (best loss has been reached 5 times)
            loss = np.round(losses[-1], 10)
            if loss < best_loss:
                best_loss_counter = 0
                best_loss = loss
            elif loss == best_loss:
                best_loss_counter += 1

            if best_loss_counter >= 5:
                break

        if losses[-1] < 0:
            warnings.warn(f"Reached negative loss {losses[-1]}")
        return losses

    def implied_sigma_mu(
        self, suffix: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    def forward(self) -> torch.Tensor:
        """Return a value proportional to the log likelihood."""
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
                    [sigma.index_select(0, x[1]).index_select(1, x[1]) for x in pairs],
                )
                L_sigmas = torch.linalg.cholesky(sigmas)
                sigmas_logdet = L_sigmas.diagonal(dim1=1, dim2=2).log().sum(dim=1) * 2

                for i, (observations, available) in enumerate(pairs):
                    mean_diff = mean_diffs.index_select(0, observations).index_select(
                        1, available,
                    )
                    loss_current = sigmas_logdet[i] * len(observations) + torch.trace(
                        mean_diff.mm(torch.cholesky_solve(mean_diff.t(), L_sigmas[i])),
                    )
                    loss = loss + loss_current.clamp(min=0.0)

        else:
            # maximum likelihood
            L_sigma = torch.linalg.cholesky(sigma)
            loss = L_sigma.diagonal().log().sum() * 2 + torch.trace(
                torch.cholesky_solve(self.sample_covariance, L_sigma),
            )

        return loss
