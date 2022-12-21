# pylint: disable=import-outside-toplevel

''' Module contains code for initialization of MS segmentation framework '''

import logging
import numpy as np
from sklearn.decomposition import PCA
from msiplib.segmentation import rebuild_segment_numbering

def initialize_segmentation(image, k, valid_mask, args, gt=None, seed=42):
    '''
        Initialize segmentation for application of MS functional to hyperspectral image segmentation

        :param image: the image that is to be segmented
        :param k: number of clusters that are sought
        :param valid_mask: mask of pixels to be input to kmeans
    '''
    logger = logging.getLogger('init')

    # TODO: Changed order of dimensionality reduction for segmentation and kmeans. Kmeans has own dim.red!

    if (image.shape[-1] > 3) and args['reduce_dim_init']:
        # TODO: adapt logging for all methods
        max_kmeans_channels = 0.95            # use so many channels such that 95% of the variance are explained
        pca = PCA(n_components=max_kmeans_channels)
        pca.fit(image.reshape((-1, image.shape[-1])))
        init_kmeans_dim = pca.n_components_.astype(np.uint8)
        init_kmeans_explained_var = pca.explained_variance_ratio_.sum()#.astype(image.dtype)
        logger.info('Reduce dimensionality for initialization with PCA.')
        logger.info('Dimensions initialization runs on: %s', init_kmeans_dim)
        logger.info('Initialization - Accumulated explained variance: %s', init_kmeans_explained_var)
        reshapedimage = pca.transform(image.reshape((-1, image.shape[-1])))
    else:
        logger.info('Initialization runs on full dimension: %s', image.shape[-1])
        reshapedimage = np.reshape(image, newshape=(-1, image.shape[-1]))

    logger.info('Initialization method: %s', args['init'])

    # K-Means
    if args['init'] == 'kmeans':
        from sklearn.cluster import KMeans
        init = 'k-means++'
        n_init = 10
        max_iter = 300
        tol = 1e-04
        logger.info('Kmeans initialization: %s', init)
        logger.info('Kmeans number of initializations: %s', n_init)
        logger.info('Kmeans maximum number of iterations: %s', max_iter)
        logger.info('Kmeans tolerance: %s', tol)
        logger.info('Seed for randomness: %s', seed)
        kmeans = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol,
                        random_state=seed).fit(reshapedimage[valid_mask.reshape(-1)])
        seg_valid = kmeans.labels_.astype(np.uint8)

    # Random
    elif args['init'] == 'random':
        logger.info('Seed for randomness: %s', seed)
        rng = np.random.default_rng(seed)
        seg_valid = rng.integers(low=0, high=k, size=valid_mask.sum(), dtype=np.uint8)

    # DBSCAN
    elif args['init'] == 'dbscan':
        from sklearn.cluster import DBSCAN
        eps = 0.5
        min_samples = 5
        metric = 'euclidean'
        p = 2
        logger.info('DBSCAN epsilon: %s', eps)
        logger.info('DBSCAN minimum number of samples in a neighborhood: %s', min_samples)
        logger.info('DBSCAN metric: %s', metric)
        logger.info('DBSCAN exponent of Minkowski metric: %s', p)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, p=p)
        seg_valid = dbscan.fit_predict(reshapedimage[valid_mask.reshape(-1)])

    # Gaussian mixture model
    elif args['init'] == 'gmm':
        from sklearn.mixture import GaussianMixture
        covariance_type = 'full'
        n_init = 1
        init_params = 'kmeans'
        logger.info('GMM covariance type: %s', covariance_type)
        logger.info('GMM number of initializations: %s', n_init)
        logger.info('GMM initialization: %s', init_params)
        logger.info('Seed for randomness: %s', seed)
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init,
                             init_params=init_params, random_state=seed)
        seg_valid = gm.fit_predict(reshapedimage[valid_mask.reshape(-1)])

    # Hierarchical clustering
    elif args['init'] == 'hierarchical':
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.image import grid_to_graph
        con = grid_to_graph(n_x=image.shape[0], n_y=image.shape[1], n_z=1, mask=valid_mask)
        # con = kneighbors_graph(reshapedimage[valid_mask.reshape(-1)], n_neighbors=3, mode='connectivity',
        #                        metric='minkowski', p=2, include_self=False)
        affinity = 'euclidean'
        connectivity = 'Pixel-to-pixel connections'
        linkage = 'ward'
        logger.info('Hierarchical affinity: %s', affinity)
        logger.info('Hierarchical connectivity: %s', connectivity)
        logger.info('Hierarchical linkage: %s', linkage)
        model = AgglomerativeClustering(n_clusters=k, affinity=affinity, connectivity=con, linkage=linkage)
        seg_valid = model.fit_predict(reshapedimage[valid_mask.reshape(-1)])

    # OPTICS
    elif args['init'] == 'optics':
        from sklearn.cluster import OPTICS
        min_samples = 5
        max_eps = 0.75
        metric = 'minkowski'
        p = 2
        cluster_method = 'xi'
        xi = 0.05
        pred_correction = True
        min_cluster_size = None
        leaf_size = 30
        logger.info('OPTICS minimum number of samples in a neighborhood: %s', min_samples)
        logger.info('OPTICS maximum distance of two pixels in neighborhood: %s', max_eps)
        logger.info('OPTICS metric: %s', metric)
        logger.info('OPTICS exponent of Minkowski metric: %s', p)
        logger.info('OPTICS cluster method: %s', cluster_method)
        if cluster_method == 'xi':
            logger.info('OPTICS minimum steepness on the reachability plot: %s', xi)
            logger.info('OPTICS predecessor correction: %s', pred_correction)
            logger.info('OPTICS minimum cluster size: %s', min_cluster_size)
        logger.info('OPTICS leaf size: %s', min_cluster_size)
        optics = OPTICS(min_samples=min_samples, max_eps=max_eps, metric=metric, p=p, cluster_method=cluster_method,
                        eps=max_eps, xi=xi, predecessor_correction=pred_correction, min_cluster_size=min_cluster_size,
                        leaf_size=leaf_size)
        seg_valid = optics.fit_predict(reshapedimage[valid_mask.reshape(-1)])

    # Birch
    elif args['init'] == 'birch':
        from sklearn.cluster import Birch
        threshold = 0.5
        branching_fac = 50
        logger.info('Birch threshold: %s', threshold)
        logger.info('Birch branching factor: %s', branching_fac)
        brc = Birch(threshold=threshold, branching_factor=branching_fac, n_clusters=k, compute_labels=True, copy=True)
        seg_valid = brc.fit_predict(reshapedimage[valid_mask.reshape(-1)])

    # Ground truth
    elif args['init'] == 'gt':
        if gt is None:
            raise Exception('No ground truth available.')
        else:
            seg_valid = gt[valid_mask]
            if args['ignore_label'] is not None:
                labels = np.unique(gt[gt != args['ignore_label']])
                ignored_shape = seg_valid[seg_valid == args['ignore_label']].shape
                logger.info('Seed for randomness: %s', seed)
                logger.info('Fill the pixels with ignore label with random valid labels.')
                rng = np.random.default_rng(seed)
                inds = rng.integers(low=0, high=k, size=ignored_shape, dtype=np.uint8)
                seg_valid[seg_valid == args['ignore_label']] = labels[inds]
            seg_valid = rebuild_segment_numbering(seg_valid)

    # insert clustering labels into a new segmentation mask
    # segmentation = np.zeros(image.shape[:-1], dtype=np.uint8)
    logger.info('Labels of non-valid pixels: random')
    rng = np.random.default_rng(seed=seed)
    logger.info('Seed for randomness: %s', seed)
    segmentation = rng.integers(low=0, high=k, size=image.shape[:-1], dtype=np.uint8)
    segmentation[valid_mask] = seg_valid
    segmentation = segmentation.reshape(image.shape[:-1])

    return segmentation
