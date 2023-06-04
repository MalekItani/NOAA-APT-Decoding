import utils
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square
import scipy.signal as signal
import argparse
import os
import glob
import librosa
import cv2
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot_fast
import torch
import matplotlib.cm as cm
import matplotlib

matplotlib.use('TkAgg')

device = 'cpu'


def get_cloud_mask(img: np.ndarray) -> np.ndarray:
    cloud_mask = ((img > 200) * 255).astype(np.uint8)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5,5))
    cloud_mask = cv2.dilate(cloud_mask, kernel=ellipse_kernel, iterations=2)
    
    return cloud_mask



class ImageCombiner():
    def __init__(self,
                 nms_radius = 4,
                 keypoint_threshold = 0.005,
                 max_keypoints = -1,
                 superglue = 'outdoor',
                 sinkhorn_iterations = 20,
                 match_threshold = 0.2) -> None:
        config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': superglue,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(device)

        self.keys = ['keypoints', 'scores', 'descriptors']

        self.current_image = None

    def extract_keypoints(self, img: np.ndarray, keypoint_mask=None) -> dict:
        frame = frame2tensor(img, device=device)
        data = self.matching.superpoint({'image': frame})
        data['image'] = frame
        
        if keypoint_mask is not None:
            kp_int = np.round(data['keypoints'][0].cpu().numpy()).astype(np.int32)
            mask = (keypoint_mask[kp_int[:, 1], kp_int[:, 0]] == 0)
                        
            data['keypoints'][0] = data['keypoints'][0][mask, :]
            data['scores'] = (data['scores'][0][mask], )
            data['descriptors'][0] = data['descriptors'][0][:, mask]

        return data

    def compute_homography(self, img1: np.ndarray, img2: np.ndarray, debug: bool=False) -> np.ndarray:
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(img1, None)
        kp2, desc2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(desc1,desc2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

        return M

    def _compute_homography(self, img1: np.ndarray, img2: np.ndarray, debug: bool=False) -> np.ndarray:
        with torch.no_grad():
            # Get masked keypoints for image 1
            cloud_mask1 = get_cloud_mask(img1)
            keypoints = self.extract_keypoints(img1, keypoint_mask=cloud_mask1)
            keys = keypoints.keys()
            keypoints1 = {k+'0': keypoints[k] for k in keys}

            # Get masked keypoints for image 2
            cloud_mask2 = get_cloud_mask(img2)
            keypoints = self.extract_keypoints(img2, keypoint_mask=cloud_mask2)
            keypoints2 = {k+'1': keypoints[k] for k in keys}
            
            # Match keypoints using neural network
            pred = self.matching({**keypoints1, **keypoints2})
            kpts0 = keypoints1['keypoints0'][0].cpu().numpy()
            kpts1 = keypoints2['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()

            # Get valid matches
            valid = (matches > -1) & (confidence > 0.8)
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]

            # Plot keypoints if debugging
            if debug:
                k_thresh = self.matching.superpoint.config['keypoint_threshold']
                m_thresh = self.matching.superglue.config['match_threshold']
                small_text = [
                    'Keypoint Threshold: {:.4f}'.format(k_thresh),
                    'Match Threshold: {:.2f}'.format(m_thresh)
                ]
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0))
                ]
                color = cm.jet(confidence[valid])                
                out = make_matching_plot_fast(
                        img1, img2, kpts0, kpts1, mkpts0, mkpts1, color, text,
                        path=None, show_keypoints=True, small_text=small_text)
                plt.imshow(out, cmap='gray', vmin=0, vmax=255)
                plt.title('Keypoints')
                plt.show()

            homography, mask = cv2.findHomography(mkpts1, mkpts0)

            return homography

    def feed(self, img: np.ndarray, debug=False) -> np.ndarray:
        A = self.current_image
        B = img
        
        if A is None:
            self.current_image = img
        else:
            # Compute homography between two images
            homography = self.compute_homography(A, img, debug=debug)

            # Warp input image to current image
            dst = cv2.warpPerspective(img, homography, A.T.shape)

            if debug:
                debug_img = np.concatenate([A, dst], axis=1)
                
                plt.imshow(debug_img, cmap='gray', vmin=0, vmax=255)
                plt.show()

            # Add images (where non-zero)
            res = np.zeros_like(dst).astype(np.float32)
            mask = (dst > 0) * 1 + (A > 0) * 1
            res[dst > 0] += dst[dst > 0]
            res[A > 0] += A[A  > 0]
            res[mask > 0] = res[mask > 0] / mask[mask > 0]
            res = res.astype(np.uint8)
            
            # Update current image
            self.current_image = res

    def draw_borders(self, ref_img: np.ndarray, border_mask: np.ndarray, debug: bool=False):
        homography = self.compute_homography(self.current_image, ref_img, debug=debug)

        # Warp input image to current image
        warped_borders = cv2.warpPerspective(border_mask, homography, self.current_image.T.shape)
        
        # Apply borders on top of current image
        self.current_image = (warped_borders > 0) * warped_borders + (warped_borders == 0) * self.current_image

    def get(self):
        return self.current_image


def main(args):
    imgs = []
    for img_path in glob.glob(os.path.join(args.path, '*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        imgs.append(utils.cut_image(img))

    reference_img = cv2.flip(cv2.imread(args.noaa_reference_path, cv2.IMREAD_GRAYSCALE), -1)
    reference_img_no_borders = cv2.flip(cv2.imread(args.noaa_reference_path[:-4] + '_no_borders.png', cv2.IMREAD_GRAYSCALE), -1)
    border_mask = ((reference_img >= 252) * 255).astype(np.uint8)
    
    combiner = ImageCombiner()

    img = None
    for i in range(len(imgs)):
        combiner.feed(imgs[i], debug=0)

    combiner.draw_borders(reference_img_no_borders, border_mask, 1)

    img = combiner.get()
    img = cv2.flip(img, -1) # Undo rotations
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("noaa_reference_path")
    args = parser.parse_args()
    main(args)