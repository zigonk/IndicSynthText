import numpy as np
import synth_utils as su
import cv2


class TextRegions(object):
    """
    Get region from segmentation which are good for placing
    text.
    """
    minWidth = 30  # px
    minHeight = 30  # px
    minAspect = 0.3  # w > 0.3*h
    maxAspect = 7
    minArea = 100  # number of pix
    pArea = 0.60  # area_obj/area_minrect >= 0.6

    # RANSAC planar fitting params:
    dist_thresh = 0.10  # m
    num_inlier = 90
    ransac_fit_trials = 100
    min_z_projection = 0.25

    minW = 20

    @staticmethod
    def filter_rectified(mask):
        """
        mask : 1 where "ON", 0 where "OFF"
        """
        wx = np.median(np.sum(mask, axis=0))
        wy = np.median(np.sum(mask, axis=1))
        return wx > TextRegions.minW and wy > TextRegions.minW

    @staticmethod
    def get_hw(pt, return_rot=False):
        pt = pt.copy()
        R = su.unrotate2d(pt)
        mu = np.median(pt, axis=0)
        pt = (pt-mu[None, :]).dot(R.T) + mu[None, :]
        h, w = np.max(pt, axis=0) - np.min(pt, axis=0)
        if return_rot:
            return h, w, R
        return h, w

    @staticmethod
    def filter(seg, area, label):
        """
        Apply the filter.
        The final list is ranked by area.
        """
        good = label[area > TextRegions.minArea]
        area = area[area > TextRegions.minArea]
        filt, R = [], []
        for idx, i in enumerate(good):
            mask = seg == i
            xs, ys = np.where(mask)

            coords = np.c_[xs, ys].astype('float32')
            rect = cv2.minAreaRect(coords)
            box = np.array(cv2.boxPoints(rect))
            h, w, rot = TextRegions.get_hw(box, return_rot=True)

            f = (h > TextRegions.minHeight
                 and w > TextRegions.minWidth
                 and TextRegions.minAspect < w/h < TextRegions.maxAspect
                 and area[idx]/w*h > TextRegions.pArea)
            filt.append(f)
            R.append(rot)

        # filter bad regions:
        filt = np.array(filt)
        area = area[filt]
        R = [R[i] for i in range(len(R)) if filt[i]]

        # sort the regions based on areas:
        aidx = np.argsort(-area)
        good = good[filt][aidx]
        R = [R[i] for i in aidx]
        filter_info = {'label': good, 'rot': R, 'area': area[aidx]}
        return filter_info

    @staticmethod
    def sample_grid_neighbours(mask, nsample, step=3):
        """
        Given a HxW binary mask, sample 4 neighbours on the grid,
        in the cardinal directions, STEP pixels away.
        """
        if 2*step >= min(mask.shape[:2]):
            return  # None

        y_m, x_m = np.where(mask)
        mask_idx = np.zeros_like(mask, 'int32')
        for i in range(len(y_m)):
            mask_idx[y_m[i], x_m[i]] = i

        xp, xn = np.zeros_like(mask), np.zeros_like(mask)
        yp, yn = np.zeros_like(mask), np.zeros_like(mask)
        xp[:, :-2*step] = mask[:, 2*step:]
        xn[:, 2*step:] = mask[:, :-2*step]
        yp[:-2*step, :] = mask[2*step:, :]
        yn[2*step:, :] = mask[:-2*step, :]
        valid = mask & xp & xn & yp & yn

        ys, xs = np.where(valid)
        N = len(ys)
        if N == 0:  # no valid pixels in mask:
            return  # None
        nsample = min(nsample, N)
        idx = np.random.choice(N, nsample, replace=False)
        # generate neighborhood matrix:
        # (1+4)x2xNsample (2 for y,x)
        xs, ys = xs[idx], ys[idx]
        s = step
        X = np.transpose(np.c_[xs, xs+s, xs+s, xs-s, xs-s]
                         [:, :, None], (1, 2, 0))
        Y = np.transpose(np.c_[ys, ys+s, ys-s, ys+s, ys-s]
                         [:, :, None], (1, 2, 0))
        sample_idx = np.concatenate([Y, X], axis=1)
        mask_nn_idx = np.zeros((5, sample_idx.shape[-1]), 'int32')
        for i in range(sample_idx.shape[-1]):
            mask_nn_idx[:, i] = mask_idx[sample_idx[:, :, i]
                                         [:, 0], sample_idx[:, :, i][:, 1]]
        return mask_nn_idx

    @staticmethod
    def filter_depth(xyz, seg, regions):
        plane_info = {'label': [],
                      'coeff': [],
                      'support': [],
                      'rot': [],
                      'area': []}
        for idx, l in enumerate(regions['label']):
            mask = seg == l
            pt_sample = TextRegions.sample_grid_neighbours(
                mask, TextRegions.ransac_fit_trials, step=3)
            if pt_sample is None:
                continue  # not enough points for RANSAC
            # get-depths
            pt = xyz[mask]
            plane_model = su.isplanar(pt, pt_sample,
                                      TextRegions.dist_thresh,
                                      TextRegions.num_inlier,
                                      TextRegions.min_z_projection)
            if plane_model is not None:
                plane_coeff = plane_model[0]
                if np.abs(plane_coeff[2]) > TextRegions.min_z_projection:
                    plane_info['label'].append(l)
                    plane_info['coeff'].append(plane_model[0])
                    plane_info['support'].append(plane_model[1])
                    plane_info['rot'].append(regions['rot'][idx])
                    plane_info['area'].append(regions['area'][idx])

        return plane_info

    @staticmethod
    def get_regions(xyz, seg, area, label):
        regions = TextRegions.filter(seg, area, label)
        # fit plane to text-regions:
        regions = TextRegions.filter_depth(xyz, seg, regions)
        return regions