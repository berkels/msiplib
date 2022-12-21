# pylint: disable=import-outside-toplevel

''' Module contains functions for dimensionality reduction in MS segmentation framework '''

import logging

def band_selection(image, method, param, seed=42):
    '''
        applies a band selection method to extract relevant spectral bands

        Input:
            image: image containing all available spectral bands
            method: method applied to select relevant bands
            param: parameter of band selection method
            seed: seed to make results reproducible
    '''
    logger = logging.getLogger('init')
    logger.info('Band selection method: %s', method)

def reduce_number_features(image, dim_red_method, num_features=0.999, trim_proportion=0.49, seed=42):
    '''
        reduces number of feature channels of image to speed up computation and reduce noise

        :param image: image of which spectral dimension is to be reduced
        :param dim_red_method: method with which spectral dimensionality is reduced
        :param num_features: number of features to be kept.
                             if in (0, 1) and PCA is used, keep so many features
                             such that explained variance is at least as large as chosen number.
        :param trim_proportion: if tga is used, determine trim proportion
    '''

    # check if argument num_features is integer. if so, copy it to n_features
    # if num_features is a float, check if it actually represents an integer. if so, cast it into an integer
    if isinstance(num_features, int):
        n_features = num_features
    else:
        if num_features.is_integer():
            n_features = int(num_features)
        else:
            n_features = num_features

    # if number of features is larger than number of channels, reduce to number of channels
    if n_features > image.shape[-1]:
        n_features = image.shape[-1]

    logger = logging.getLogger('init')
    logger.info('Dimensionality reduction method: %s', dim_red_method)
    logger.info('Number of features to keep: %s', n_features)

    # all methods except MNF need input rolled out as a vector
    if dim_red_method != 'mnf':
        im_reshaped = image.reshape((-1, image.shape[-1]))

    # PCA
    if dim_red_method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_features)
        im_reduced = pca.fit_transform(im_reshaped)
        logger.info('Kept features: %s', pca.n_components_)
        logger.info('Accumulated explained variance: %s', pca.explained_variance_ratio_.sum())

    # MNF
    elif dim_red_method == 'mnf':
        from spectral import calc_stats, mnf, noise_from_diffs
        sig = calc_stats(image)
        noise_dir = 'lowerright'
        logger.info('Noise estimation direction: %s', noise_dir)
        noise = noise_from_diffs(image, direction=noise_dir)
        mnfr = mnf(sig, noise)
        im_reduced = mnfr.reduce(image, num=n_features).astype(image.dtype)
        logger.info('Number of kept NAPCs: %s', n_features)
        logger.info('Minimum SNR of kept NAPCs: %s', min(mnfr.napc.eigenvalues[:n_features]) - 1)

    # T-SNE
    elif dim_red_method == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = 30
        early_ex = 12
        lr = 200
        metric = 'euclidean'
        logger.info('Perplexity: %s', perplexity)
        logger.info('Early exaggeration: %s', early_ex)
        logger.info('Learning rate: %s', lr)
        logger.info('Metric: %s', metric)
        logger.info('Seed for randomness: %s', seed)
        tsne = TSNE(n_components=n_features, perplexity=perplexity, early_exaggeration=early_ex,
                    learning_rate=lr, metric=metric, random_state=seed)
        im_reduced = tsne.fit_transform(im_reshaped)

    # Umap
    elif dim_red_method == 'umap':
        from umap import UMAP
        n_neighbors = 16
        min_dist = 0.1
        metric = 'euclidean'
        logger.info('Number of neighbors: %s', n_neighbors)
        logger.info('Minimum distance: %s', min_dist)
        logger.info('Metric: %s', metric)
        logger.info('Seed for randomness: %s', seed)
        reducer = UMAP(n_components=n_features, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                       random_state=seed)
        im_reduced = reducer.fit_transform(im_reshaped)

    # IsoMap
    elif dim_red_method == 'isomap':
        from sklearn.manifold import Isomap
        n_neighbors = 8
        metric = 'minkowski'
        p = 2
        logger.info('Number of neighbors: %s', n_neighbors)
        logger.info('Metric: %s', metric)
        logger.info('Order of Minkowski metric: %s', p)
        reducer = Isomap(n_components=n_features, n_neighbors=n_neighbors, metric=metric, p=p)
        im_reduced = reducer.fit_transform(im_reshaped)



    # ICA
    elif dim_red_method == "ica" :
        from sklearn.decomposition import  FastICA
        fun = 'logcosh'
        logger.info('Function: %s', fun)
        logger.info('Seed for randomness: %s', seed)
        ica = FastICA(n_components=n_features, fun=fun, random_state=seed)
        im_reduced = ica.fit_transform(im_reshaped)

    return im_reduced.reshape((image.shape[0], image.shape[1], im_reduced.shape[-1]))
