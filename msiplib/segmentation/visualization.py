import numpy as np
from msiplib.segmentation import create_segmentation_colormap


def plot_RGB_feature_distributions(image, labels, ignore_label=None, size=1.5, cmap=None, means=None, pcs=None,
                                   std_devs=None):
    ''' plots the feature distribution of an RGB image given segmentation labels '''
    import plotly.graph_objects as go

    diff_labels = np.unique(labels)
    num_segments = diff_labels.shape[0]

    if cmap is None:
        cmap = (255 * create_segmentation_colormap())[:len(diff_labels)]
    else:
        if num_segments > cmap.shape[0]:
            for i in range(num_segments - cmap.shape[0]):
                new_col = 1 / 2 * (cmap[i] + cmap[i + 1])
                cmap = np.concatenate((cmap, new_col[np.newaxis]), axis=0)

    fig = go.Figure()

    # if colormap contains grayscale values, convert the map to an rgb map
    if cmap.ndim == 1:
        cmap = np.transpose(np.broadcast_to(cmap, (3, cmap.shape[0])))

    # plot pixels with ignore label
    if ignore_label is not None:
        vectors = image[labels == ignore_label]
        color = 'rgb({}, {}, {})'.format(cmap[ignore_label, 0], cmap[ignore_label, 1], cmap[ignore_label, 2])
        fig.add_trace(go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                                   name=('{}'.format('Pxs with ignore label')), mode='markers',
                                   marker_color=(color)))
        # remove color used by ignored pixels
        cmap = cmap[diff_labels != ignore_label]
        diff_labels = diff_labels[diff_labels != ignore_label]

    # add a trace for every segment
    for i in range(len(diff_labels)):
        vectors = image[labels == diff_labels[i]]
        color = 'rgb({}, {}, {})'.format(cmap[i, 0], cmap[i, 1], cmap[i, 2])
        fig.add_trace(go.Scatter3d(x=vectors[:, 0], y=vectors[:, 1], z=vectors[:, 2],
                                   name=('Segment {}'.format(i + 1)), mode='markers',
                                   marker_color=(color)))

    fig.update_traces(mode='markers', marker_line_width=2, marker_size=size)

    if means is not None:
        for i in range(len(diff_labels)):
            color = 'rgb({}, {}, {})'.format(cmap[i, 0], cmap[i, 1], cmap[i, 2])
            fig.add_trace(go.Scatter3d(x=[means[i, 0]], y=[means[i, 1]], z=[means[i, 2]],
                                    name=('Mean of segment {}'.format(i + 1)), marker_color=(color)))

    # if pcs is not None and std_devs is not None:
    #     for i in range(len(diff_labels)):
    #         for j in range(3):
    #             fig = fig.add_trace(go.Cone(
    #                                 x=[means[i, 0]],
    #                                 y=[means[i, 1]],
    #                                 z=[means[i, 2]],
    #                                 u=[std_devs[i, j] * pcs[i, j, 0]],
    #                                 v=[std_devs[i, j] * pcs[i, j, 1]],
    #                                 w=[std_devs[i, j] * pcs[i, j, 2]],
    #                                 sizemode="absolute",
    #                                 sizeref=2,
    #                                 anchor="tail",
    #                                 name=('PC {} of segment {}'.format(j + 1, i + 1))))

    # fig.update_layout(
    #     scene=dict(domain_x=[0, 1],
    #                camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

    # Set options common to all traces
    fig.update_layout(title='Feature distribution', yaxis_zeroline=False, xaxis_zeroline=False)

    fig.show()
