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

        self.last_data = None
        self.last_image = None

    def feed(self, img1: np.ndarray) -> np.ndarray:
        print(img1.shape)
        with torch.no_grad():
            frame = frame2tensor(img1, device=device)
            if self.last_data is None:
                self.last_data = self.matching.superpoint({'image': frame})
                self.last_data = {k+'0': self.last_data[k] for k in self.keys}
                self.last_data['image0'] = frame
            else:
                pred = self.matching({**self.last_data, 'image1': frame})
                kpts0 = self.last_data['keypoints0'][0].cpu().numpy()
                kpts1 = pred['keypoints1'][0].cpu().numpy()
                matches = pred['matches0'][0].cpu().numpy()
                confidence = pred['matching_scores0'][0].cpu().numpy()

                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]

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
                        self.last_image, img1, kpts0, kpts1, mkpts0, mkpts1, color, text,
                        path=None, show_keypoints=True, small_text=small_text)
                
                plt.imshow(out, cmap='gray', vmin=0, vmax=255)
                plt.show()

                homography, mask = cv2.findHomography(mkpts0, mkpts1)

                print('Before', img1.shape)
                # homography_inverse = np.linalg.inv(homography)
                dst = cv2.warpPerspective(img1, homography, self.last_image.T.shape)

                print('Warped', dst.shape)
                print('Last image', self.last_image.shape)

                debug = np.concatenate([self.last_image, dst], axis=1)
                plt.imshow(debug, cmap='gray', vmin=0, vmax=255)
                plt.show()

                # res = np.zeros_like(dst).astype(np.float32)
                # mask = (dst > 0) & (self.last_image > 0)
                
                # res[dst > 0] += dst[dst > 0]
                # res[self.last_image > 0] += self.last_image[self.last_image  > 0]

                # print('res', res.shape)
                # print('mask', (mask>0).shape)
                # res[mask > 0] = res[mask > 0] / mask[mask > 0]
                # res = res.astype(np.uint8)

                res = ((dst.astype(np.float32) + self.last_image.astype(np.float32))/2).astype(np.uint8)

                plt.imshow(res, cmap='gray', vmin=0, vmax=255)
                plt.show()

            self.last_image = img1

def main(args):
    imgs = []
    for img_path in glob.glob(os.path.join(args.path, '*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        imgs.append(utils.cut_image(img))

    combiner = ImageCombiner()

    img = None
    for i in range(len(imgs)):
        img = combiner.feed(imgs[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    main(args)