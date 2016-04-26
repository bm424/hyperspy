
import numpy as np
import abc


def guess_prototypes(x, q):
    """Produces initial guesses for the cluster centers.

    In this implementation the guesses are distributed in a random

    Parameters
    ----------
    x : (n_samples, n_variables), array_like
        Unlabeled object data.
    q : int
        Number of clusters to find.
    Returns
    -------
    c : (q, n_variables), ndarray
        Cluster centers.

    """
    c = 2 * np.std(x) * np.random.random((q, x.shape[1]))
    return c


class AlternatingOptimization(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, x, n_clusters=2, max_iter=200, tol=1e-4, **kwargs):
        """Alternating optimization (A/O) algorithm for clustering.

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        n_clusters : int
            Number of clusters to find.
        kwargs : Arguments passed to subclasses

        """

        # Store unlabeled data
        self.X = x

        # Set parameters
        self.Q = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.params = kwargs

        # Set initial values
        self.D = None  # Distance matrix
        self.C = guess_prototypes(x, n_clusters)  # Cluster centers
        self.W = 0  # Cluster weights
        self.U = None  # Cluster memberships
        self.J = np.infty
        self.started = False

    @abc.abstractmethod
    def d_squared(self, x, c):
        """Calculates the squared distance between data 'x' and centers 'c'.

        Parameters
        ----------
        x : ndarray
            (n_samples, n_variables)
            Unlabeled object data.
        c : ndarray
            (n_clusters, n_variables)
            Cluster centers.

        Returns
        -------
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        """
        return

    @abc.abstractmethod
    def u(self, d):
        """Calculates cluster membership based on distance to centers 'd'.

        Parameters
        ----------
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        Returns
        -------
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.

        """
        return

    @abc.abstractmethod
    def c(self, x, u):
        """Calculates cluster centers based on memberships 'u'.

        Parameters
        ----------
        x : ndarray
            (n_samples, n_variables)
            Unlabeled object data.
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.

        Returns
        -------
        c : ndarray
            (n_clusters, n_variables)
            Cluster centers.

        """
        return

    @abc.abstractmethod
    def j(self, u, d):
        """Calculates the objective function based on memberships 'u'.

        Parameters
        ----------
        u : ndarray
            (n_clusters, n_samples)
            Data point memberships.
        d : ndarray
            (n_clusters, n_samples)
            Distance metric.

        Returns
        -------
        j : float
            Value of the objective function for this partition.

        """
        return

    def start(self):
        """First routine to be implemented by the subclasses before iteration.

        """
        pass

    def __iter__(self):
        return self

    def next(self):
        """Update distances, memberships, and cluster centers.

        Returns
        -------
        self.J : The new value of the minimisation function.

        """
        if self.started is False:
            self.start()
            self.started = True
        self.D = self.d_squared(self.X, self.C)  # Recalculate distance matrix
        self.U = self.u(self.D)  # Update memberships with the old centers
        self.C = self.c(self.X, self.U)  # Update centers with new memberships
        self.J = self.j(self.U, self.D)
        return self.J

    def optimize(self):
        for i in range(self.max_iter):
            if self.J < self.next() + self.tol:
                break
        return self

    @property
    def model(self):
        return np.dot(self.U.T, self.C)

    @property
    def entropy(self):
        return -np.sum(self.U * np.log(self.U))


class CMeans(AlternatingOptimization):

    def __init__(self, *args, **kwargs):
        super(CMeans, self).__init__(*args, **kwargs)
        if 'm' in kwargs:
            self.m = kwargs['m']
        else:
            self.m = 2

    @staticmethod
    def diff(x, c):
        """Returns a difference array.

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        c : (n_clusters, n_variables) array_like
            Cluster centers.

        Returns
        -------
        ndarray
            The vector differences between each data point and each cluster
            center.
            (n_clusters, n_samples, n_variables)

        """
        return x - c[:, np.newaxis]

    def e(self, x, c):
        """Covariance of data assigned to each cluster

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        c : (n_clusters, n_variables) array_like
            Cluster centers.

        Returns
        -------
        ndarray
            The (modified) covariance matrix of the data assigned to each
            cluster.

        """
        q, p = c.shape
        if self.U is None:
            return (np.eye(p)[..., np.newaxis] * np.ones((p, q))).T
        v = self.diff(x, c)
        u = self.g(self.U)
        outer = np.einsum('...i,...j->...ij', v, v)
        es = np.einsum('...i,...ijk', u, outer) / np.sum(u, axis=1)[
            ..., np.newaxis, np.newaxis]
        p = x.shape[1]
        return es / (np.linalg.det(es) ** (1. / p))[..., np.newaxis, np.newaxis]

    def d_squared(self, x, c):
        """Returns a distance matrix.

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        c : (n_clusters, n_variables) array_like
            Cluster centers.

        Returns
        -------
        d : ndarray
            The square of the Euclidian distance between a data point and the
            cluster center.

        """
        v = self.diff(x, c)
        d = np.sum(v ** 2, axis=2)
        return d

    def g(self, u):
        """Fuzzification operator.

        Parameters
        ----------
        u : (n_clusters, n_samples) array_like
            Cluster memberships.

        Returns
        -------
        g : (n_clusters, n_samples) array_like
            Fuzzified memberships.

        """
        g = u ** self.m
        return g


class Hard(CMeans):

    def u(self, d):
        """Calculates cluster memberships.

        Parameters
        ----------
        d : (n_clusters, n_samples) array_like
            Distance matrix.

        Returns
        -------
        u : (n_clusters, n_samples) array_like
            The membership of each data point to each cluster.

        """
        p = d.shape[0]
        u = np.arange(p)[:, np.newaxis] == np.argmin(d, axis=0)
        return u

    def c(self, x, u):
        """Calculate cluster centers based on membership and data

        Parameters
        ----------
        x : (n_samples, n_variables) array_like
            Unlabeled object data.
        u : (n_clusters, n_samples) array_like
            Cluster memberships.

        Returns
        -------
        c : (n_clusters, n_variables) array_like
            The new position of the cluster centers.

        """
        return np.dot(u, x) / np.sum(u, axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(u * d)


class PossibilisticFuzzy(CMeans):

    def start(self):
        initializer = ProbabilisticFuzzy(
            self.X, self.Q, self.max_iter, self.tol, **self.params)
        initializer.optimize()
        self.D = initializer.D
        self.U = initializer.U
        self.C = initializer.C
        self.W = self._guess_weights(self.U, self.D)

    def _guess_weights(self, u, d):
        w = np.sum(self.g(u) * d, axis=1) / np.sum(self.g(u), axis=1)
        return w

    def u(self, d):
        return (1. + (d / self.W[:, np.newaxis]) ** (1. / (self.m - 1))) ** -1.

    def c(self, x, u):
        return np.dot(self.g(u), x) / np.sum(self.g(u), axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(self.g(u) * d) \
               + np.sum(self.W * np.sum(self.g(1. - u), axis=1))


class ProbabilisticFuzzy(CMeans):

    def u(self, d):
        return d ** (-2. / (self.m - 1)) / np.sum(d ** -2. / (self.m - 1),
                                                  axis=0)

    def c(self, x, u):
        return np.dot(self.g(u), x) / np.sum(self.g(u), axis=1)[..., np.newaxis]

    def j(self, u, d):
        return np.sum(self.g(u) * d)


class GustafsonKessel(ProbabilisticFuzzy):
    def d_squared(self, x, c):
        V = self.diff(x, c)
        E = self.e(x, c)
        pre = np.einsum('...ij,...jk', V, np.linalg.inv(E))
        return np.sum(pre * V, axis=2)


class FCV(ProbabilisticFuzzy):
    def d_squared(self, x, c):
        V = self.diff(x, c)
        self.B = self.b(x, c)
        return np.sum(V ** 2, axis=2) - np.sum(
            np.einsum('...ij,...jk', V, self.B), axis=2)

    def b(self, x, c):
        eigenvalues, eigenvectors = np.linalg.eig(self.e(x, c))
        idx = eigenvalues.argsort()[:, ::-1]
        sorted_evectors = np.array(
            [eigenvectors[p][:, idx[p]] for p in range(len(c))])
        return sorted_evectors[:, :, :self.r]


class RousseeuwTrauwaertKaufman(ProbabilisticFuzzy):
    def __init__(self, *args, **kwargs):
        super(RousseeuwTrauwaertKaufman, self).__init__(*args, **kwargs)
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']

    def g(self, u):
        return self.alpha * u + (1. - self.alpha) * u ** 2

    def u(self, d):
        beta = (1. - self.alpha) / (1. + self.alpha)
        if self.U is None:
            u_init = d ** -1. / np.sum(d ** -1., axis=0)
            return u_init
        else:
            u_old = self.U
            for i, (d, u) in enumerate(zip(d.T, u_old.T)):
                condition = u > 0
                q = np.sum(condition)
                u_new = ((1. + (q - 1.) * beta) / (
                    d * np.sum(1. / d[condition])) - beta) / (1. - beta)
                u_new[u_new < 0] = 0
                u_old[:, i] = u_new
            return u_old