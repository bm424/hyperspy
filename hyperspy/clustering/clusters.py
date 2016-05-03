import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from .tools import try_again
from .algorithms import ProbabilisticFuzzy


class ClusterTools:

    def cluster(self, n_clusters=2, algorithm=ProbabilisticFuzzy,
                use_learning_results=True, attempts=1, signal_mask=None,
                navigation_mask=None, **kwargs):
        """Assigns the data to automatically-inferred clusters.

        Parameters
        ----------
        n_clusters : int
            The number of cluster prototypes to find.
        algorithm : clustering.algorithms.AlternatingOptimisation
            The c-means algorithm used to optimise the cluster
            membership and centers.
        use_learning_results : bool
            if True, use decomposition learning results as
            the cluster data. Otherwise use data directly.
        attempts : int
            The number of times to restart the algorithm.
        kwargs : Arguments to be passed to the cluster algorithm (see specific
            documentation for details).

        """
        with self.unfolded():
            self._use_learning_results = use_learning_results

            if hasattr(signal_mask, 'ravel'):
                signal_mask = signal_mask.ravel()
            elif hasattr(navigation_mask, 'ravel'):
                navigation_mask = navigation_mask.ravel()

            # Transform the None masks in slices to get the right behaviour
            if navigation_mask is None:
                navigation_mask = slice(None)
            else:
                navigation_mask = ~navigation_mask
            if signal_mask is None:
                signal_mask = slice(None)
            else:
                signal_mask = ~signal_mask

            if use_learning_results is False:
                self.x = self.data
            elif use_learning_results is True:
                self.x = self.learning_results.loadings
            else:
                raise ValueError(
                    "'_use_learning_results' must be True or False")

            target = self.learning_results
            result = try_again(self.x[:, signal_mask][navigation_mask, :],
                               n_clusters, algorithm, attempts, **kwargs)
            target.centers = result.C
            target.membership = result.U
            target.cluster_algorithm = result

            # Set the pixels that were not processed to nan
            if not isinstance(signal_mask, slice):
                # Store the (inverted, as inputed) signal mask
                target.signal_mask = ~signal_mask.reshape(
                    self.axes_manager._signal_shape_in_array)
                centers = np.zeros((self.x.shape[-1], target.centers.shape[1]))
                centers[signal_mask, :] = target.centers
                centers[~signal_mask, :] = np.nan
                target.centers = centers
            if not isinstance(navigation_mask, slice):
                # Store the (inverted, as inputed) navigation mask
                target.navigation_mask = ~navigation_mask.reshape(
                    self.axes_manager._navigation_shape_in_array)
                membership = np.zeros((target.membership.shape[0],
                                       self.x.shape[0]))
                membership[:, navigation_mask] = target.membership
                membership[:, ~navigation_mask] = np.nan
                target.membership = membership

    def undecompose(self, points):
        """

        Parameters
        ----------
        points : ndarray
            (n_points, n_features)

        Returns
        -------
        ndarray:
            (n_points, n_original_features)
            The undecomposed points.

        """
        if self._use_learning_results is True:
            return np.dot(points, self.learning_results.factors.T).T
        else:
            return points.T

    def undecompose_centers(self):
        """Undo the decomposition on the derived cluster centers.

        Because clustering is most often performed on a
        reduced-dimensionality dataset, for visualisation it is necessary to
        re-project the derived centers into the original high-dimension space.

        Returns
        -------
        centers : ndarray
            (n_centers, n_original_features)
            The derived cluster centers, reprojected into the original
            high-dimensional space.

        """
        return self.undecompose(self.learning_results.centers)

    def get_cluster_centers(self):
        """Returns the cluster centers as a signal."""
        centers = self.undecompose_centers()
        return self._get_factors(centers)

    def plot_cluster_centers(self, **kwargs):
        """Plots the cluster centers in their original dimensionality."""
        c = self.undecompose_centers()
        return self._plot_factors_or_pchars(c, comp_label="Cluster center",
                                            **kwargs)

    def get_cluster_memberships(self):
        """Returns the cluster memberships as a signal."""
        return self._get_loadings(self.learning_results.membership.T)

    def plot_cluster_memberships(self, with_centers=False, per_row=3, **kwargs):
        """Plots cluster memberships."""
        u = self.learning_results.membership
        if with_centers is True:
            factors = self.undecompose_centers()
        else:
            factors = None
        return self._plot_loadings(u, comp_label="Cluster membership",
                                   with_factors=with_centers,
                                   factors=factors, per_row=per_row, **kwargs)

    def plot_cluster_results(self):
        """Plots the results of the clustering (memberships and centers)."""
        centers = self.get_cluster_centers()
        memberships = self.get_cluster_memberships()
        centers.axes_manager._axes[0] = memberships.axes_manager._axes[0]
        centers.plot()
        memberships.plot()
        return centers, memberships

    @property
    def partition(self):
        """The 'hardened' clusters.

        The partition is a list of length 'n_clusters', each element of which is
        a subset of the data whose membership is closest to one for that
        cluster.

        """
        self.unfold()
        x = self.learning_results.loadings
        u = self.learning_results.membership
        partition = [x[m == np.max(u, axis=0)] for m in u]
        self.fold()
        return partition

    def plot2d(self, indices=(0, 1), size_multiplier=30., size_offset=1.):
        """Creates a plot of the data projected on to two dimensions.

        Parameters
        ----------
        indices : tuple of int
            The indices representing the dimensions to plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis object containing the plot.

        Notes
        -----
        Must be called after 'cluster()' as 'learning_results' are used. This
        to be fixed in a future version.

        """

        try:
            x = self.learning_results.loadings
        except AttributeError:
            self.unfold()
            x = self.data
            self.fold()
        u = self.learning_results.membership
        c = self.learning_results.centers
        try:
            q = len(u)
        except TypeError:
            self.unfold()
            q = 1
            u = np.ones((1, self.axes_manager.navigation_size))
            c = self.mean().data.reshape((1, -1))
            self.fold()

        a = indices[0]
        b = indices[1]

        fig = plt.figure()

        ax = fig.add_subplot(111)
        s_list = []
        cmap = plt.get_cmap('Set1')
        colors = [cmap(i) for i in np.linspace(0, 1, len(u))]
        for i, u_i in enumerate(u):
            select = np.max(u, axis=0) == u_i
            s = ax.scatter(x[:, a][select], x[:, b][select], marker='o',
                           s=size_multiplier * (u_i[select] - size_offset / q),
                           c=colors[i])
            ax.scatter(c[i][a], c[i][b], marker='o', s=80, c=colors[i])
            s_list.append(s)
        plt.title(self.learning_results.cluster_algorithm)
        plt.legend(s_list, range(q))
        return ax

    def plot3d(self, indices=(0, 1, 2), size_multiplier=30., size_offset=0.,
               navigation_mask=None):
        """Creates a plot of the data projected into three dimensions.

        Different clusters are plotted in different colours, and the
        membership is represented by varying the size of the points.

        Parameters
        ----------
        indices : tuple of int
            The indices of the dimensions to be plotted.
        size_offset : float
            Offsets the size of the points for v
        size_multiplier : float
            Multiplies the size of the data points.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis object containing the plot.

        Notes
        -----
        Must be called after 'cluster()' as 'learning_results' are used. This
        to be fixed in a future version.

        """
        try:
            x = self.learning_results.loadings
        except AttributeError:
            self.unfold()
            x = self.data
            self.fold()
        u = self.learning_results.membership
        c = self.learning_results.centers

        try:
            q = len(u)
        except TypeError:
            self.unfold()
            q = 1
            u = np.ones((1, self.axes_manager.navigation_size))
            c = self.mean().data.reshape((1, -1))
            self.fold()

        i0 = indices[0]
        i1 = indices[1]
        i2 = indices[2]

        if x.shape[1] < 3:
            raise ValueError("Data dimension is insufficient for 3-d plotting.")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        s_list = []
        cmap = plt.get_cmap('Set1')
        colors = [cmap(i) for i in np.linspace(0, 1, len(u))]
        for i, u_i in enumerate(u):
            select = np.max(u, axis=0) == u_i
            s = ax.scatter(xs=x[:, i0][select], ys=x[:, i1][select],
                           zs=x[:, i2][select], marker='o',
                           s=size_multiplier * (u_i[select] - size_offset / q),
                           c=colors[i])
            ax.scatter(c[i][i0], c[i][i1], c[i][i2], marker='o', s=80,
                       c=colors[i])
            s_list.append(s)
        plt.title(self.learning_results.cluster_algorithm.__class__.__name__)
        plt.legend(s_list, range(q))
        return ax

    def regression(self):
        self.unfold()
        x = self.data
        c = self.undecompose_centers()
        self.fold()
        return linalg.lstsq(x.T, c)[0].T
