# ComfyUI custom node: TwoStrongestColors (fixed)
# Save as: ComfyUI/custom_nodes/two_strongest_colors.py
# Restart ComfyUI.

import math
import torch
import numpy as np

CSS3_COLORS = {
    "black": (0, 0, 0), "dimgray": (105,105,105), "gray": (128,128,128),
    "darkgray": (169,169,169), "silver": (192,192,192), "gainsboro": (220,220,220),
    "whitesmoke": (245,245,245), "white": (255,255,255),
    "maroon": (128,0,0), "darkred": (139,0,0), "firebrick": (178,34,34),
    "crimson": (220,20,60), "red": (255,0,0), "tomato": (255,99,71), "orangered": (255,69,0),
    "saddlebrown": (139,69,19), "sienna": (160,82,45), "chocolate": (210,105,30),
    "peru": (205,133,63), "sandybrown": (244,164,96), "burlywood": (222,184,135), "tan": (210,180,140),
    "darkorange": (255,140,0), "orange": (255,165,0), "gold": (255,215,0),
    "olive": (128,128,0), "olivedrab": (107,142,35), "darkolivegreen": (85,107,47),
    "yellowgreen": (154,205,50), "chartreuse": (127,255,0), "greenyellow": (173,255,47),
    "darkgreen": (0,100,0), "green": (0,128,0), "seagreen": (46,139,87),
    "mediumseagreen": (60,179,113), "limegreen": (50,205,50), "lime": (0,255,0),
    "forestgreen": (34,139,34), "springgreen": (0,255,127),
    "teal": (0,128,128), "darkcyan": (0,139,139), "lightseagreen": (32,178,170),
    "turquoise": (64,224,208), "mediumturquoise": (72,209,204), "aquamarine": (127,255,212),
    "cyan": (0,255,255), "lightcyan": (224,255,255),
    "navy": (0,0,128), "darkblue": (0,0,139), "mediumblue": (0,0,205),
    "blue": (0,0,255), "royalblue": (65,105,225), "dodgerblue": (30,144,255),
    "deepskyblue": (0,191,255), "lightskyblue": (135,206,250), "skyblue": (135,206,235),
    "steelblue": (70,130,180), "lightsteelblue": (176,196,222),
    "indigo": (75,0,130), "purple": (128,0,128), "darkmagenta": (139,0,139),
    "magenta": (255,0,255), "orchid": (218,112,214), "mediumorchid": (186,85,211),
    "plum": (221,160,221), "violet": (238,130,238),
    "deeppink": (255,20,147), "hotpink": (255,105,180), "palevioletred": (219,112,147),
    "pink": (255,192,203), "lightpink": (255,182,193),
    "brown": (165,42,42), "rosybrown": (188,143,143), "indianred": (205,92,92),
    "salmon": (250,128,114), "darksalmon": (233,150,122), "lightsalmon": (255,160,122),
    "khaki": (240,230,140), "darkkhaki": (189,183,107), "beige": (245,245,220),
    "cornsilk": (255,248,220), "wheat": (245,222,179), "bisque": (255,228,196),
    "moccasin": (255,228,181), "navajowhite": (255,222,173), "peachpuff": (255,218,185)
}

def _pivot_rgb(u): return np.where(u > 0.04045, ((u + 0.055)/1.055) ** 2.4, u / 12.92)
def _rgb_to_xyz(rgb):
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    r,g,b = _pivot_rgb(r), _pivot_rgb(g), _pivot_rgb(b)
    x = r*0.4124564 + g*0.3575761 + b*0.1804375
    y = r*0.2126729 + g*0.7151522 + b*0.0721750
    z = r*0.0193339 + g*0.1191920 + b*0.9503041
    return np.stack([x,y,z], -1)
def _pivot_xyz(t):
    eps, kappa = 216/24389, 24389/27
    return np.where(t > eps, np.cbrt(t), (kappa*t + 16)/116)
def _rgb_to_lab(rgb_uint8):
    rgb = rgb_uint8.astype(np.float32)/255.0
    xyz = _rgb_to_xyz(rgb)
    Xn,Yn,Zn = 0.95047, 1.0, 1.08883
    f = _pivot_xyz(np.stack([xyz[...,0]/Xn, xyz[...,1]/Yn, xyz[...,2]/Zn], -1))
    L = 116*f[...,1] - 16
    a = 500*(f[...,0]-f[...,1])
    b = 200*(f[...,1]-f[...,2])
    return np.stack([L,a,b], -1)

_css_names = list(CSS3_COLORS.keys())
_css_rgbs  = np.array([CSS3_COLORS[n] for n in _css_names], dtype=np.uint8)
_css_labs  = _rgb_to_lab(_css_rgbs)

def _nearest_css_color_name(rgb_uint8_triplet):
    lab = _rgb_to_lab(np.array(rgb_uint8_triplet, dtype=np.uint8)[None,:])
    d2 = np.sum((_css_labs - lab)**2, axis=1)
    return _css_names[int(np.argmin(d2))]

def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=np.float32)
    i0 = rng.integers(0, n); centroids[0] = X[i0]
    closest = np.sum((X - centroids[0])**2, axis=1)
    for c in range(1, k):
        probs = closest / (closest.sum() + 1e-12)
        idx = rng.choice(n, p=probs); centroids[c] = X[idx]
        dist_sq = np.sum((X - centroids[c])**2, axis=1)
        closest = np.minimum(closest, dist_sq)
    return centroids

def _kmeans(X, k=6, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    centroids = _kmeans_pp_init(X, k, rng)
    for _ in range(iters):
        d = np.sum((X[:,None,:] - centroids[None,:,:])**2, axis=2)
        labels = np.argmin(d, axis=1)
        for j in range(k):
            m = labels == j
            if np.any(m): centroids[j] = X[m].mean(axis=0)
            else: centroids[j] = X[rng.integers(0, X.shape[0])]
    d = np.sum((X[:,None,:] - centroids[None,:,:])**2, axis=2)
    labels = np.argmin(d, axis=1)
    return centroids, labels

class TwoStrongestColors:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ignore_near_white": ("BOOLEAN", {"default": True}),
                "ignore_near_black": ("BOOLEAN", {"default": True}),  # better default for black BG
                "whiteness_threshold": ("FLOAT", {"default": 0.92, "min": 0.5, "max": 1.0, "step": 0.01}),
                "blackness_threshold": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "k_clusters": ("INT", {"default": 6, "min": 2, "max": 12}),
                "sample_limit": ("INT", {"default": 200000, "min": 1000, "max": 2000000}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("color_1_name", "color_2_name")
    FUNCTION = "analyze"
    CATEGORY = "Color/Analysis"

    def _prepare_pixels(self, img_tensor, alpha_threshold, ignore_near_white, ignore_near_black,
                        whiteness_threshold, blackness_threshold, sample_limit, seed):
        if img_tensor is None:
            return np.zeros((0,3), dtype=np.uint8)

        if img_tensor.dim() != 4:
            raise ValueError("Expected IMAGE tensor with shape [B,H,W,C].")

        B,H,W,C = img_tensor.shape
        if C < 3:
            raise ValueError("Image must have at least 3 channels (RGB).")
        if C > 4:  # trim extra channels defensively
            img_tensor = img_tensor[...,:4]; C = 4

        img = img_tensor.detach().cpu().numpy()
        img = np.clip(img, 0.0, 1.0)

        if C == 4:
            rgb = (img[...,:3] * 255.0).astype(np.uint8).reshape(-1,3)
            a   = img[...,3].reshape(-1)
            alpha_mask = a > alpha_threshold
        else:
            rgb = (img[...,:3] * 255.0).astype(np.uint8).reshape(-1,3)
            alpha_mask = np.ones((rgb.shape[0],), dtype=bool)

        rgb_norm = rgb.astype(np.float32) / 255.0
        intensity = rgb_norm.mean(axis=1)

        mask = alpha_mask.copy()
        if ignore_near_white: mask &= (intensity < whiteness_threshold)
        if ignore_near_black: mask &= (intensity > blackness_threshold)

        rgb = rgb[mask]
        if rgb.size == 0:
            return rgb

        if rgb.shape[0] > sample_limit:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, rgb.shape[0], size=sample_limit)
            rgb = rgb[idx]
        return rgb

    def analyze(self, image, alpha_threshold=0.05, ignore_near_white=True, ignore_near_black=True,
                whiteness_threshold=0.92, blackness_threshold=0.08, k_clusters=6,
                sample_limit=200000, random_seed=0):

        # Accept single tensor OR list of tensors
        tensors = []
        if isinstance(image, list):
            tensors = [t for t in image if torch.is_tensor(t)]
        elif torch.is_tensor(image):
            tensors = [image]
        else:
            return ("unknown", "unknown")

        if len(tensors) == 0:
            return ("unknown", "unknown")

        # Concatenate batches if possible
        try:
            img_tensor = torch.cat(tensors, dim=0)  # [B,H,W,C]
        except Exception:
            img_tensor = tensors[0]

        # First pass with provided filters
        pixels = self._prepare_pixels(
            img_tensor, alpha_threshold, ignore_near_white, ignore_near_black,
            whiteness_threshold, blackness_threshold, sample_limit, random_seed
        )

        # Retry without filtering if empty
        if pixels.size == 0:
            pixels = self._prepare_pixels(
                img_tensor, 0.0, False, False, 1.0, 0.0, sample_limit, random_seed
            )
            if pixels.size == 0:
                return ("unknown", "unknown")

        X = pixels.astype(np.float32)
        k = int(max(2, min(k_clusters, X.shape[0] // 10 if X.shape[0] >= 20 else 2)))
        centroids, labels = _kmeans(X, k=k, iters=15, seed=random_seed)

        counts = np.bincount(labels, minlength=centroids.shape[0])
        order = np.argsort(-counts)

        names, used = [], set()
        for idx in order:
            name = _nearest_css_color_name(np.clip(np.round(centroids[idx]), 0, 255).astype(np.uint8))
            if name not in used:
                names.append(name); used.add(name)
            if len(names) == 2:
                break

        if   len(names) == 0: return ("unknown", "unknown")
        elif len(names) == 1: return (names[0], names[0])
        else:                 return (names[0], names[1])

NODE_CLASS_MAPPINGS = {"TwoStrongestColors": TwoStrongestColors}
NODE_DISPLAY_NAME_MAPPINGS = {"TwoStrongestColors": "Two Strongest Colors (Names)"}
