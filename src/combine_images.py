import utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import glob
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

def find_ocean(img: np.ndarray):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ocean_mask = (((hsv_img[..., 0] >= 95) & (hsv_img[..., 0] <= 120) & (hsv_img[..., 2] <= 70))).astype(np.uint8)    
    
    ocean_mask = cv2.medianBlur(ocean_mask, ksize=5)
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5,5))
    ocean_mask = cv2.morphologyEx((ocean_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, kernel=ellipse_kernel, iterations=1)
    ocean_mask = cv2.morphologyEx((ocean_mask * 255).astype(np.uint8), cv2.MORPH_DILATE, kernel=ellipse_kernel, iterations=11)

    return ocean_mask


class ImageCombiner():
    def __init__(self,
                 nms_radius = 4,
                 keypoint_threshold = 0.001,
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
        self.ref_img = None
        self.borders = None
        self.coastlines = None
        self.imgs = []
        self.warped_imgs = []

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

    def _compute_homography(self, img1: np.ndarray, img2: np.ndarray, debug: bool=False) -> np.ndarray:
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

        M, mask = cv2.findHomography(src_pts, dst_pts)

        return M

    def compute_homography(self, img1: np.ndarray, img2: np.ndarray, debug: bool=False, conf=0.8, blur_images=False, clouds=True) -> np.ndarray:
        with torch.no_grad():
            # Get masked keypoints for image 1
            cloud_mask1 = get_cloud_mask(img1) * clouds
            if blur_images:
                img1 = cv2.medianBlur(img1, 3)
            keypoints = self.extract_keypoints(img1, keypoint_mask=cloud_mask1)
            keys = keypoints.keys()
            keypoints1 = {k+'0': keypoints[k] for k in keys}

            # Get masked keypoints for image 2
            cloud_mask2 = get_cloud_mask(img2) * clouds
            if blur_images:
                img2 = cv2.medianBlur(img2, 3)
            keypoints = self.extract_keypoints(img2, keypoint_mask=cloud_mask2)
            keypoints2 = {k+'1': keypoints[k] for k in keys}
            
            # Match keypoints using neural network
            pred = self.matching({**keypoints1, **keypoints2})
            kpts0 = keypoints1['keypoints0'][0].cpu().numpy()
            kpts1 = keypoints2['keypoints1'][0].cpu().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().numpy()

            # Get valid matches
            valid = (matches > -1) & (confidence > conf)
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

            if len(mkpts1) < 4:
                print("Couldn't find homography, skipping...")
                return None

            homography, mask = cv2.findHomography(mkpts1, mkpts0)

            return homography

    def feed(self, img: np.ndarray, debug=False) -> np.ndarray:
        A = self.current_image

        self.imgs.append(img)
        
        if A is None:
            self.current_image = img
            self.warped_imgs.append(img)
        else:
            # Compute homography between two images
            homography = self.compute_homography(A, img, debug=debug, blur_images=0)

            if homography is None:
                return

            # Warp input image to current image
            dst = cv2.warpPerspective(img, homography, A.T.shape)

            if debug:
                debug_img = np.concatenate([A, dst], axis=1)
                
                plt.imshow(debug_img, cmap='gray', vmin=0, vmax=255)
                plt.show()

            self.warped_imgs.append(dst)

            self.current_image = self.get()

    def get(self):
        # Add images (where non-zero)
        res = np.zeros_like(self.warped_imgs[0]).astype(np.float32)
        
        denom_mask = np.zeros_like(res)
        for img in self.warped_imgs:
            res += img
            denom_mask += img > 0
        
        res[denom_mask > 0] = res[denom_mask > 0] / denom_mask[denom_mask > 0]
        res = res.astype(np.uint8)

        return res

    def draw_coastlines(self,
                        img: np.ndarray,
                        ref_img_gray: np.ndarray,
                        ref_img_gray_no_borders: np.ndarray,
                        coastline_imgs: list,
                        debug: bool = False) -> np.ndarray:
        
        border_mask = ((ref_img_gray >= 252) * 255).astype(np.uint8)
        
        # Add borders
        homography = self.compute_homography(img, ref_img_gray_no_borders, debug=debug)
        if homography is None:
            return

        # Warp input image to current image
        borders = cv2.warpPerspective(border_mask, homography, img.T.shape)

        coastlines = None
        for wborders, woborders in coastline_imgs:
            # Add coastline
            if borders is None:
                print("No borders found, cannot add coastline!")
                return
            
            ref_w_borders_gray = cv2.cvtColor(wborders, cv2.COLOR_BGR2GRAY)
            border_mask = ((ref_w_borders_gray >= 252) * 255).astype(np.uint8)
            
            
            homography = self.compute_homography(borders//5, border_mask//5, debug=debug, clouds=False)
            if homography is None:
                return

            # Add new borders
            warped_borders = cv2.warpPerspective(border_mask, homography, borders.T.shape)

            if debug:
                debug_img = np.concatenate([warped_borders, borders], axis=1)
                
                plt.imshow(debug_img, cmap='gray', vmin=0, vmax=255)
                plt.show()

            borders = np.maximum(borders, warped_borders)
            
            # Get oceans
            ocean_mask = find_ocean(woborders)

            # Add new coaslines
            ocean_mask_warped = cv2.warpPerspective(ocean_mask, homography, borders.T.shape)

            if coastlines is None:
               coastlines = (ocean_mask_warped * borders)
            else:
                coastlines = np.maximum(coastlines, (ocean_mask_warped * borders))

            if debug:            
                plt.imshow(ocean_mask_warped, cmap='gray', vmin=0, vmax=255)
                plt.show()
        
        # Add coastlines
        ret = ((coastlines > 0) * 255 + (coastlines == 0) * img).astype(np.uint8)

        return ret
    
    def plot_noise_resilience(self,
                              img: np.ndarray,
                              ref_img_gray: np.ndarray,
                              ref_img_gray_no_borders: np.ndarray,
                              coastline_imgs: list,
                              debug: bool = False, 
                              title: str = 'unknown'):
        
        SNRs = [20, 10, 5, 0]
        img_0to1 = img/255
        img_power = 10*np.log10(np.var(img_0to1))

        fig, axes = plt.subplots(nrows=1, ncols=len(SNRs))
        fig.set_size_inches(16, 4)
        for i, snr in enumerate(SNRs):
            noise_power = 10**((img_power-snr)/20)
            n0 = np.random.randn(*img.shape)*noise_power
            
            # Add noise
            noisy_img = np.round(np.clip(img_0to1 + n0, 0, 1) * 255).astype(np.uint8)
            measured_noise_power = 10*np.log10(np.var(n0))
            realized_snr = img_power - measured_noise_power
            print(f"Realized {realized_snr}, expected {snr}")

            coastline_img = self.draw_coastlines(noisy_img, ref_img_gray, ref_img_gray_no_borders, coastline_imgs, debug=debug)
            
            if coastline_img is not None:
                coastline_img = cv2.flip(coastline_img, -1) # Undo rotations
                axes[i].imshow(coastline_img, cmap='gray', vmin=0, vmax=255)
            else:
                noisy_img = cv2.flip(noisy_img, -1) # Undo rotations
                axes[i].imshow(noisy_img, cmap='gray', vmin=0, vmax=255)
            
            axes[i].set_title(f'Noise level: {snr}dB')
            label = chr(ord('A') + i)
            axes[i].set_xlabel(f'({label})')
        
        fig.suptitle(title.capitalize())
        plt.savefig(f'output/Project2/All/{title}_resilience.png')
        plt.clf()
        plt.close(fig)
        

def combine_and_draw_coastlines(args):
    imgs = []
    for img_path in glob.glob(os.path.join(args.path, '*.png')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        title = os.path.basename(img_path)[:-4]
        imgs.append((utils.cut_image(img), title))

    imgs = sorted(imgs, key = lambda x: -x[0].shape[0]) # Sort images by decreasing height
    titles = [x[1] for x in imgs]
    imgs = [x[0] for x in imgs]

    reference_img = cv2.flip(cv2.imread(args.noaa_reference_path, cv2.IMREAD_COLOR), -1)
    reference_img_no_borders = cv2.flip(cv2.imread(args.noaa_reference_path[:-4] + '_no_borders.png', cv2.IMREAD_COLOR), -1)
    
    reference_img_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    reference_img_no_borders_gray = cv2.cvtColor(reference_img_no_borders, cv2.COLOR_BGR2GRAY)

    combiner = ImageCombiner()

    coastline_imgs = []
    for img_path in glob.glob(os.path.join(args.noaa_coastlines, '*_borders.png')):
        print(img_path)
        wborders = cv2.flip(cv2.imread(img_path[:-len('_no_borders.png')] + '.png', cv2.IMREAD_COLOR), -1)
        woborders = cv2.flip(cv2.imread(img_path, cv2.IMREAD_COLOR), -1)
        coastline_imgs.append((wborders, woborders))

    img = None
    quadrahelix_plotted = dipole_plotted = False
    for i in range(len(imgs)):
        print("Now processing", titles[i])
        combiner.feed(imgs[i], debug=0)
        ret = combiner.draw_coastlines(imgs[i], reference_img_gray, reference_img_no_borders_gray, coastline_imgs)
        
        if ret is not None:
            ret = cv2.flip(ret, -1) # Undo rotations
            
            if 'quadra' in titles[i] and (not quadrahelix_plotted):
                quadrahelix_plotted = True
                
                combiner.plot_noise_resilience(imgs[i], 
                                               reference_img_gray,
                                               reference_img_no_borders_gray,
                                               coastline_imgs,
                                               title=titles[i] + " noise resilience")

            if 'dipole' in titles[i] and (not dipole_plotted):
                dipole_plotted = True
                
                combiner.plot_noise_resilience(imgs[i], 
                                               reference_img_gray,
                                               reference_img_no_borders_gray,
                                               coastline_imgs,
                                               title=titles[i] + " noise resilience")
        else:
            ret = imgs[i]
        
        plt.imshow(ret, cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i].capitalize())
        plt.savefig(f'output/Project2/All/{titles[i]}.png')
        plt.close()

    img = combiner.get()

    img = combiner.draw_coastlines(img, reference_img_gray, reference_img_no_borders_gray, coastline_imgs)

    combiner.plot_noise_resilience(img, reference_img_gray, reference_img_no_borders_gray, coastline_imgs, title='combined')

    img = cv2.flip(img, -1) # Undo rotations
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title("Coherently combined")
    plt.savefig(f'output/Project2/All/sum_combined.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("noaa_reference_path")
    parser.add_argument("noaa_coastlines")
    args = parser.parse_args()
    combine_and_draw_coastlines(args)